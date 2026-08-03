"""
Per-epoch GPU-memory profiler for the MDMM x SimpleRouter training loop.

WHY: on the Gautschi L40S this run leaks ~170 MB/epoch -- proven by proportion,
not guessed: it survived ~253 epochs with the whole 47 GB card and ~123 epochs
with a 20 GB cap (half the memory -> half the epochs). The identical code does
NOT leak on the AF A100 over 5000 epochs. Two guesses (async allocator, memory
cap) already failed to fix it, so this measures instead.

METHOD: bisect the per-epoch work by callback. Each mode runs in its OWN process
(memory state must not carry across modes) and logs current/peak GPU bytes, host
RSS and the live Python object count every epoch. The slope of `gpu_current`
against epoch is the leak rate; whichever mode flattens it contains the culprit.

  full      production callbacks (router logger + MDMM logger + checkpoints)
  no_mdmm   drop MDMMStateLogger  -- the per-epoch FULL-BATCH (5000-event,
                                     ~517 MB) deterministic forward pass, my
                                     prime suspect
  no_router drop SimpleRouterLogger (per-epoch mu/theta pulls to numpy)
  bare      CSVLogger only -- the floor: if this still leaks it is fit() itself

  python profile_leak.py --mode full --epochs 40 --out <dir>
"""
import os, sys, csv, gc, argparse, resource

HELPERS = os.environ.get(
    "SMARTPIX_HELPERS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "two_bit_optimization_helpers"))
sys.path.insert(0, HELPERS)

import numpy as np
import tensorflow as tf

_MEM_MB = os.environ.get("SMARTPIX_GPU_MEM_MB")
for g in tf.config.list_physical_devices("GPU"):
    try:
        if _MEM_MB:
            tf.config.set_logical_device_configuration(
                g, [tf.config.LogicalDeviceConfiguration(memory_limit=int(_MEM_MB))])
        else:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

from prepare_tfrecords import load_tfrecords
from train import create_model
from loss import custom_loss
from mdmm import MDMM, MinCorrConstraint

BASE = os.environ.get(
    "SMARTPIX_DATA_BASE",
    "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d")
TFR = os.environ.get("SMARTPIX_TFR", os.path.join(BASE, "TFR_files_all101_noise_contained_discovery"))
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)
OUT_COLS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
LAB_COLS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}
MB = 1024.0 ** 2


class MemProbe(tf.keras.callbacks.Callback):
    def __init__(s, path, mode):
        super().__init__(); s.path = path; s.mode = mode
        with open(s.path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "gpu_current_mb", "gpu_peak_mb", "host_rss_mb", "gc_objects"])
    def on_epoch_end(s, epoch, logs=None):
        cur = pk = -1.0
        try:
            mi = tf.config.experimental.get_memory_info("GPU:0")
            cur, pk = mi["current"] / MB, mi["peak"] / MB
        except Exception:
            pass
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        n_obj = len(gc.get_objects())
        with open(s.path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{cur:.1f}", f"{pk:.1f}", f"{rss:.1f}", n_obj])
        print(f"[{s.mode}] ep {epoch}: gpu_cur {cur:.0f} MB  peak {pk:.0f} MB  "
              f"rss {rss:.0f} MB  objs {n_obj}", flush=True)


class _Sub(tf.keras.callbacks.Callback):
    """Runs one isolated statement per epoch, for the statement-level bisect."""
    def __init__(s, fn):
        super().__init__(); s.fn = fn
    def on_epoch_end(s, epoch, logs=None):
        _ = s.fn()


class RouterLogger(tf.keras.callbacks.Callback):
    """Mirrors the production SimpleRouterLogger's per-epoch work."""
    def on_epoch_end(s, epoch, logs=None):
        r = s.model.get_layer("simple_router_output")
        _ = r.theta.numpy(); _ = r.mu_numpy(); _ = r.visits.numpy(); _ = r.selected_indices()


class MdmmLogger(tf.keras.callbacks.Callback):
    """Mirrors the production MDMMStateLogger: a per-epoch FULL-BATCH
    deterministic forward pass over the cached 5000-event diagnostic batch."""
    def __init__(s, inner, x, y, constraints):
        super().__init__(); s.inner = inner; s.x = x; s.y = y; s.cons = constraints
    def on_epoch_end(s, epoch, logs=None):
        # PROF_TENSOR_INPUT=1 feeds a pre-converted tf.Tensor instead of numpy --
        # the A/B that confirms the 122 MB/epoch leak is the numpy handoff.
        preds = s.inner(s.x, training=False).numpy()
        for name, col in OUT_COLS.items():
            p = preds[:, col]; t = s.y[:, LAB_COLS[name]]
            _ = float(np.mean((p - p.mean()) * (t - t.mean())) / (p.std() * t.std() + 1e-6))
        _ = [float(c.lmbda.numpy()) for c in s.cons]


def main():
    ap = argparse.ArgumentParser()
    # The sub-* modes bisect INSIDE the leaking diagnostic callback, one
    # statement at a time, after the callback-level bisect narrowed it to there
    # and the numpy-vs-tensor A/B came back identical (+122.1 MB/epoch both).
    ap.add_argument("--mode", default="full",
                    choices=["full", "no_mdmm", "no_router", "bare", "mdmm_tensor",
                             "sub_fwd", "sub_fwd_numpy", "sub_lambda"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tf.random.set_seed(7); np.random.seed(7)
    tg, vg = load_tfrecords(os.path.join(TFR, "TFR_train"), os.path.join(TFR, "TFR_test"),
                            noise=-1, seed=42)
    cx, cy = vg[0]
    cx = np.asarray(cx); cy = np.asarray(cy)

    vit = create_model("ViT_Max_SimpleRouter", timeslices=101, soft_quantize_layer=True,
                       initial_thresholds=[30., 60., 120.], threshold_offset=0.0,
                       initial_levels=LEVELS)
    cons = [MinCorrConstraint(column=OUT_COLS[p], label_column=LAB_COLS[p], min_value=0.5,
                              scale=1e4, damping=1.0, name=f"corr_{p}") for p in OUT_COLS]
    model = MDMM(vit, cons, constraint_pass="primary", name="mdmm_prof")
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss=custom_loss)

    cbs = [MemProbe(os.path.join(a.out, f"mem_{a.mode}.csv"), a.mode)]
    if a.mode in ("full", "no_mdmm"):
        cbs.append(RouterLogger())
    if a.mode in ("full", "no_router"):
        cbs.append(MdmmLogger(vit, cx, cy, cons))          # numpy input (leaks)
    if a.mode == "mdmm_tensor":
        # identical diagnostic, but X pre-converted to a tf.Tensor once.
        # RESULT: leaked +122.1 MB/epoch, same as numpy -- hypothesis refuted.
        cbs.append(MdmmLogger(vit, tf.convert_to_tensor(cx), cy, cons))

    # ---- statement-level bisect of the leaking callback ----------------------
    xt = tf.convert_to_tensor(cx)
    if a.mode == "sub_fwd":
        # ONLY the forward pass; result discarded without .numpy()
        cbs.append(_Sub(lambda: vit(xt, training=False)))
    if a.mode == "sub_fwd_numpy":
        # forward pass + the device->host copy
        cbs.append(_Sub(lambda: vit(xt, training=False).numpy()))
    if a.mode == "sub_lambda":
        # ONLY the Lagrange-multiplier reads (no forward pass at all)
        cbs.append(_Sub(lambda: [float(c.lmbda.numpy()) for c in cons]))

    print(f"=== profiling mode={a.mode} epochs={a.epochs} "
          f"cap={_MEM_MB or 'growth'} ===", flush=True)
    model.fit(tg, validation_data=vg, epochs=a.epochs, callbacks=cbs, shuffle=False, verbose=0)

    # slope over the last 2/3, where startup transients have settled
    rows = list(csv.DictReader(open(os.path.join(a.out, f"mem_{a.mode}.csv"))))
    if len(rows) >= 6:
        k = len(rows) // 3
        e = np.array([int(r["epoch"]) for r in rows[k:]], float)
        m = np.array([float(r["gpu_current_mb"]) for r in rows[k:]], float)
        slope = np.polyfit(e, m, 1)[0] if len(e) > 1 else float("nan")
        print(f"RESULT mode={a.mode} leak={slope:+.1f} MB/epoch "
              f"(first {m[0]:.0f} MB -> last {m[-1]:.0f} MB)", flush=True)


main()
