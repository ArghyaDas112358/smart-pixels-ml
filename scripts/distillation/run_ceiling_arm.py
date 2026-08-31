"""O23 -- the information-ceiling ladder.

How much of the -40K NLL floor is the NETWORK and how much is the ASIC readout
budget? Four arms, differing ONLY in what the backbone is allowed to see:

  A  101 slices, full precision   the ceiling
  B    2 slices, full precision   isolates the cost of 2-bit quantization
  C  101 slices, 2-bit            isolates the cost of discarding 99 slices
  D    2 slices, 2-bit            the production model (already banked, -40,162)

Every arm uses ViT_MaxDeep -- the DEEP (256,128,64) head. Building the
full-precision arms from ViT_Max instead would silently substitute the shallow
(64,) head and understate the ceiling, which is the one direction of error that
would falsely confirm the hypothesis under test.

Arms A-C have NO router, so they also have no per-step pair sampling. Attempt 1
of O22 established that sampling was load-bearing regularization, so these arms
are genuine overfitting candidates -- and an overfit ceiling arm understates the
ceiling. Hence: early stopping on val_plain_nll with restore_best_weights, and a
divergence watch that prints the train/val gap every epoch.

  python run_ceiling_arm.py --arm A --epochs 3000 --seed 30042
"""
import os, sys, json, time, argparse
import numpy as np
R = "/work/users/das214/SmartPixels/smart-pixels-ml"
sys.path.insert(0, os.path.join(R, "two_bit_optimization_helpers"))
sys.path.insert(0, os.path.join(R, "two_bit_optimization_helpers", "models"))
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
from prepare_tfrecords import load_tfrecords
from train import create_model
import transformer_model_nonquantized as TM
from conditional_nll import custom_loss_v2

BASE = os.environ.get("SMARTPIX_BASE",
      "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_"
      "10ps_300k_convolved_to_200ps/shuffled_3d")
TFR = os.environ.get("SMARTPIX_TFR",
      os.path.join(BASE, "TFR_files_all101_noise_contained_discovery"))
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)
# the production front end, frozen -- arms B and C must reproduce D's readout
SLICES = [10, 21]
THRESHOLDS = [8.565593719482422, 20.227340698242188, 47.49168395996094]
# arm -> (n slices the backbone sees, quantize?)
ARMS = {"A": (101, False), "B": (2, False), "C": (101, True), "D": (2, True)}


class Checkpointer(tf.keras.callbacks.Callback):
    """Snapshot weights + OPTIMIZER STATE every `every` epochs so a multi-day run
    survives a process death. Weights alone are not enough: Nadam carries
    first/second-moment accumulators, and restarting without them throws the
    model off its trajectory for hundreds of epochs -- which on this model is
    the difference between reaching the deep basin and not."""
    def __init__(s, out, every=25):
        super().__init__(); s.out = out; s.every = every; s.ck = None
    def on_epoch_end(s, e, logs=None):
        if (e + 1) % s.every: return
        if s.ck is None:
            s.ck = tf.train.Checkpoint(model=s.model, optimizer=s.model.optimizer)
        s.model.save_weights(os.path.join(s.out, "last.weights.hdf5"))
        s.ck.write(os.path.join(s.out, "opt_state"))
        # write the epoch marker LAST: if we die mid-snapshot, the marker is
        # older than the weights and we simply redo a few epochs, rather than
        # resuming from a half-written checkpoint.
        json.dump({"epoch": e + 1}, open(os.path.join(s.out, "resume_state.json"), "w"))


class DivergenceWatch(tf.keras.callbacks.Callback):
    """Arms A-C have no sampler; print the train/val gap so overfitting is
    visible in the log the moment it starts rather than at the post-mortem."""
    def __init__(s, every=25): super().__init__(); s.every = every
    def on_epoch_end(s, e, logs=None):
        lg = logs or {}
        tr, va = lg.get("plain_nll"), lg.get("val_plain_nll")
        if tr is None or va is None or (e + 1) % s.every: return
        print(f"    ep {e+1:>5}  train {tr:>12,.0f}  val {va:>12,.0f}  "
              f"gap {va - tr:>+11,.0f}", flush=True)


class AbortOnStuck(tf.keras.callbacks.Callback):
    def __init__(s, thr=1e4, pat=25): super().__init__(); s.thr=thr; s.pat=pat; s.bad=0
    def on_epoch_end(s, e, logs=None):
        v = (logs or {}).get("val_loss", np.inf)
        if v > s.thr or not np.isfinite(v):
            s.bad += 1
            if s.bad >= s.pat:
                print("[AbortOnStuck] aborting", flush=True); s.model.stop_training = True
        else:
            s.bad = 0


def build(arm, seed, dropout=0.1, slice_drop=0.0, nslices=101):
    """Every arm is fed the full 101-slice tensor; the 2-slice arms gather their
    pair INSIDE the model, so the input pipeline is byte-identical across arms.

    `slice_drop` is SpatialDropout2D on the input, which drops whole channels --
    i.e. randomly hides entire time slices during training. That is the direct
    analogue of the regularization the production model gets for free from
    PairLatticeRouterLayer resampling its pair every step. Arms A-C have no
    router, and the first arm-A run overfit hard because of it: by epoch 400 the
    train/val gap had widened monotonically to +4,300 while val sat at -22,500.
    An overfit ceiling arm UNDERSTATES the ceiling, which is the one direction of
    error that would falsely confirm the hypothesis under test.
    """
    tf.keras.utils.set_random_seed(seed)
    n_in, quant = ARMS[arm]
    x_in = tf.keras.layers.Input(shape=(16, 16, 101), name="raw_input")
    x = x_in
    if n_in == 2:
        idx = SLICES
    elif nslices < 101:
        # regularly spaced over the FULL waveform, endpoints included. At
        # nslices=20 this grid is [0,5,11,16,21,26,...,95,100] -- it contains 11,
        # 21 and 26, so it can express both the discovered pair (10,21) and the
        # hand-imposed (11,26) to within one slice. A subsampled arm that could
        # not reach the production readout would not bound it.
        idx = sorted(set(np.linspace(0, 100, nslices).round().astype(int).tolist()))
        assert len(idx) == nslices, f"rounding collision: {len(idx)} unique of {nslices}"
        n_in = nslices
    else:
        idx = None
    if idx is not None:
        x = tf.keras.layers.Lambda(lambda t, i=idx: tf.gather(t, i, axis=-1),
                                   name="fixed_slices")(x)
    if slice_drop > 0:
        x = tf.keras.layers.SpatialDropout2D(slice_drop, name="slice_dropout")(x)
    if quant:
        x = TM.SoftQuantizeLayer(
            n_bits=2, initial_thresholds=THRESHOLDS, threshold_offset=0.0,
            initial_levels=LEVELS, trainable_levels=False,
            trainable_thresholds=False, initial_k=1.0, trainable_k=True,
            name="soft_quantizer_output")(x)
    out = TM._vit_backbone(x, (16, 16, n_in), 14, dropout=dropout,
                           head_dims=(256, 128, 64))
    return tf.keras.Model(x_in, out, name=f"ceiling_arm{arm}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=30042)
    ap.add_argument("--patience", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from last.weights.hdf5 + opt_state in --out")
    ap.add_argument("--ckpt-every", type=int, default=25, dest="ckpt_every")
    ap.add_argument("--nslices", type=int, default=101,
                    help="subsample the waveform to N regularly spaced slices "
                         "(arms A/C only). 101 = the full waveform.")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="transformer-block dropout (backbone default 0.1)")
    ap.add_argument("--slice-dropout", type=float, default=0.0, dest="slice_dropout",
                    help="SpatialDropout2D on the input: fraction of TIME SLICES "
                         "hidden per step. The stand-in for the router sampling "
                         "that arms A-C do not have.")
    a = ap.parse_args()
    n_in, quant = ARMS[a.arm]
    OUT = a.out or os.path.join(R, "runs", f"o23_ceiling_arm{a.arm}", f"seed_{a.seed}")
    os.makedirs(OUT, exist_ok=True)

    tg, vg = load_tfrecords(os.path.join(TFR, "TFR_train"),
                            os.path.join(TFR, "TFR_test"), noise=-1, seed=a.seed)
    print(f"[{time.strftime('%H:%M:%S')}] O23 ARM {a.arm}: "
          f"{f'{a.nslices} regularly spaced slices' if n_in == 101 else f'slices {SLICES}'}, "
          f"{'2-bit quantized at ' + str([round(t,2) for t in THRESHOLDS]) if quant else 'FULL PRECISION'}, "
          f"deep head (256,128,64)", flush=True)
    print(f"  data {TFR}  train batches {len(tg)}  val batches {len(vg)}", flush=True)

    model = build(a.arm, a.seed, dropout=a.dropout, slice_drop=a.slice_dropout,
                  nslices=a.nslices)
    print(f"  params {model.count_params():,}  input {model.input_shape}  "
          f"dropout {a.dropout}  slice_dropout {a.slice_dropout}", flush=True)

    def plain_nll(y, p):
        return custom_loss_v2(y, p)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
                  loss=custom_loss_v2, metrics=[plain_nll])

    cbs = [tf.keras.callbacks.CSVLogger(os.path.join(OUT, "history.csv"),
                                       append=os.path.exists(os.path.join(OUT, "history.csv"))),
           tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT, "best.weights.hdf5"),
                                              monitor="val_plain_nll", save_best_only=True,
                                              save_weights_only=True, mode="min"),
           DivergenceWatch(), AbortOnStuck()]
    if a.patience:
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor="val_plain_nll",
                                                    patience=a.patience, mode="min",
                                                    restore_best_weights=True, verbose=1))
    init_ep = 0
    if a.resume:
        st = os.path.join(OUT, "resume_state.json")
        w  = os.path.join(OUT, "last.weights.hdf5")
        if os.path.exists(st) and os.path.exists(w):
            init_ep = int(json.load(open(st))["epoch"])
            model.load_weights(w)
            # the optimizer's slot variables only exist after it has been built,
            # so force one no-op apply before restoring into them
            model.optimizer.build(model.trainable_variables)
            try:
                tf.train.Checkpoint(model=model, optimizer=model.optimizer).read(
                    os.path.join(OUT, "opt_state")).expect_partial()
                print(f"  RESUMED from epoch {init_ep} (weights + optimizer state)", flush=True)
            except Exception as ex:
                print(f"  RESUMED from epoch {init_ep} (weights only -- optimizer "
                      f"state not restorable: {ex})", flush=True)
        else:
            print("  --resume given but no checkpoint found; starting fresh", flush=True)

    cbs.insert(0, Checkpointer(OUT, every=a.ckpt_every))
    t0 = time.time()
    h = model.fit(tg, validation_data=vg, epochs=a.epochs, callbacks=cbs,
                  initial_epoch=init_ep, verbose=0)
    # ALWAYS save last as well: a torn val_loss line has permanently poisoned a
    # best-monitor before, freezing best.weights for thousands of epochs.
    model.save_weights(os.path.join(OUT, "last.weights.hdf5"))

    v = np.asarray(h.history.get("val_plain_nll", h.history["val_loss"]), dtype=float)
    v = v[np.isfinite(v) & (np.abs(v) < 5e6)]
    res = dict(arm=a.arm, seed=a.seed, nslices=a.nslices, dropout=a.dropout,
           resumed_from=init_ep,
           slice_dropout=a.slice_dropout, slices=(None if n_in == 101 else SLICES),
               quantized=bool(quant), thresholds=(THRESHOLDS if quant else None),
               params=int(model.count_params()), epochs_run=len(h.history["loss"]),
               best_val_plain_nll=float(v.min()) if len(v) else None,
               final_val_plain_nll=float(v[-1]) if len(v) else None,
               minutes=round((time.time() - t0) / 60, 1))
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] ARM {a.arm} DONE  "
          f"best val plain NLL {res['best_val_plain_nll']:,.0f}  "
          f"({res['epochs_run']} ep, {res['minutes']} min) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
