"""
ViT_Max Part-2 (2-bit-optimized) training, up to 1000 epochs, picking up
the charge thresholds from the partial Part-1 of the 1000-epoch run
(Part 1 was stopped at epoch 522 by the user with thresholds already in
a good regime).

Defenses against the known Part-2 plateau pathology (val_loss gets stuck
at ~1e5 due to an unlucky random init): legacy-style AbortOnStuck
callback aborts a run if val_loss stays > 1e5 for 5 epochs, and an
outer retry loop re-seeds + rebuilds the model.

Also stops cleanly once val_loss has not improved for 50 epochs
(EarlyStopping(patience=50, restore_best_weights=True)).
"""
import os, sys, json, csv, time, random, traceback
sys.path.insert(0, "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers")
import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)

from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model, train, save_performance_parquet, cleanup_models_and_generators

DATASET = "/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets"
RUNS = "/work/users/das214/SmartPixels/smart-pixels-ml/runs"
OUT = f"{RUNS}/qconv2d_max_run_full"
WEIGHTS = f"{RUNS}/weights"
PERF = f"{RUNS}/processed_parquets/test_3src/2bit_optimized"
for d in (OUT, WEIGHTS, PERF):
    os.makedirs(d, exist_ok=True)

MODEL_TYPE = "QConv2D_Max"
TIMESLICES = 2
EPOCHS = 1000
PATIENCE = 50
STUCK_THRESHOLD = 1e5     # val_loss above this means the run is stuck
STUCK_PATIENCE = 5
MAX_RETRIES = 10


class AbortOnStuck(tf.keras.callbacks.Callback):
    """Stop training early if val_loss stays > `thr` for `patience` epochs."""
    def __init__(self, threshold=1e5, patience=5):
        super().__init__()
        self.thr = threshold
        self.pat = patience
        self.bad = 0
        self.aborted = False
    def on_epoch_end(self, epoch, logs=None):
        vloss = (logs or {}).get("val_loss", float("inf"))
        if vloss > self.thr or not np.isfinite(vloss):
            self.bad += 1
            if self.bad >= self.pat:
                print(f"[AbortOnStuck] val_loss {vloss:.1f} >= {self.thr} "
                      f"for {self.pat} epochs - aborting this attempt.", flush=True)
                self.aborted = True
                self.model.stop_training = True
        else:
            self.bad = 0


def stamp(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save_hist(h, name):
    json.dump(h.history, open(f"{OUT}/{name}.json", "w"), indent=1)
    keys = list(h.history.keys())
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i in range(len(h.history[keys[0]])):
            w.writerow([i + 1] + [h.history[k][i] for k in keys])


try:
    t0 = time.time()
    thr_json = json.load(open(f"{OUT}/optimized_thresholds.json"))
    thresholds = np.array(thr_json["thresholds"], dtype=np.float32)
    levels = np.array(thr_json["levels"], dtype=np.float32)
    stamp(f"loaded thresholds from {thr_json.get('source_checkpoint')}: {thresholds.tolist()}")

    _, _, tfr_tr, tfr_val = generate_tfrecords(
        dataset_dir=DATASET, model_type=MODEL_TYPE,
        train_batch_size=5000, val_batch_size=5000,
        select_contained=False, timeslices=TIMESLICES,
        tfrecords_exist=True, seed=42,
    )
    stamp("TFRecords ready")

    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        seed = random.randint(0, 2**32 - 1)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        stamp(f"=== PART 2 attempt {attempt}/{MAX_RETRIES}, seed={seed} ===")

        tg2, vg2 = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=True,
                                  digitize_levels=levels, digitize_thresholds=thresholds,
                                  seed=seed)
        m2 = create_model(MODEL_TYPE, timeslices=TIMESLICES, soft_quantize_layer=False)

        abort_cb = AbortOnStuck(threshold=STUCK_THRESHOLD, patience=STUCK_PATIENCE)
        early_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE,
            restore_best_weights=True, verbose=1)

        try:
            ckpt2, fp2, h2 = train(
                m2, MODEL_TYPE, WEIGHTS, tg2, vg2,
                timeslices=TIMESLICES, train_type="2bit_optimized",
                epochs=EPOCHS, seed=seed, verbose=1,
                extra_callbacks=[abort_cb, early_cb],
            )
        except Exception as e:
            stamp(f"attempt {attempt} raised: {e}")
            cleanup_models_and_generators([m2, tg2, vg2])
            continue

        if abort_cb.aborted:
            stamp(f"attempt {attempt} aborted (stuck). Retrying with new seed.")
            cleanup_models_and_generators([m2, tg2, vg2])
            continue

        best_val = float(min(h2.history.get('val_loss', [float('inf')])))
        stamp(f"attempt {attempt} succeeded. best val_loss = {best_val:.2f}, "
              f"epochs run = {len(h2.history['loss'])}")
        save_hist(h2, "history_part2_2bit_optimized")
        success = True
        break

    if not success:
        raise RuntimeError(f"Part 2 failed to escape the plateau after {MAX_RETRIES} attempts.")

    stamp("=== EVALUATE: performance parquet on 3sr test split ===")
    save_performance_parquet(checkpoints=ckpt2, output_directory=PERF, test_generator=vg2,
                             model_type=MODEL_TYPE, train_type="2bit_optimized", fingerprint=fp2,
                             timeslices=TIMESLICES, soft_quantize_layer=False)

    summary = {
        "model_type": MODEL_TYPE,
        "epochs_each_part": EPOCHS,
        "part1_note": "No Part 1 for this Conv2D_Max standalone run; reused ViT_Max-derived Part-1 thresholds.",
        "part1_thresholds_source": thr_json.get("source_checkpoint"),
        "part2_final_val_loss": float(h2.history["val_loss"][-1]),
        "part2_best_val_loss": float(min(h2.history["val_loss"])),
        "part2_epochs_run": len(h2.history["loss"]),
        "part2_attempts": attempt,
        "optimized_thresholds": thr_json["thresholds"],
        "part2_checkpoints": ckpt2,
        "performance_dir": PERF,
        "total_seconds_part2": round(time.time() - t0),
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    stamp("TRAINING COMPLETE")
    print(json.dumps(summary, indent=1), flush=True)

except Exception:
    with open(f"{OUT}/FAILED.txt", "w") as f:
        f.write(traceback.format_exc())
    traceback.print_exc()
    sys.exit(1)
