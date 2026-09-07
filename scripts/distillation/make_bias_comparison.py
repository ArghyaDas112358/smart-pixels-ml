"""O22 verdict: did the per-bin zero-bias constraint flatten the residual tilt,
and what did it cost?

Three models, one figure per target:
  parent      seed 22042 last.weights  -- the state both arms warm-started FROM
  control     o22_bias_off             -- same 2,000 epochs, constraints OFF
  constrained o22_bias_on              -- same 2,000 epochs, both O22 fixes ON

The control is what makes this readable. Without it, "the bias went away" cannot
be told apart from "2,000 more epochs of ordinary training did it".

Bias is measured the way lgray reads it: mean(true - pred) per bin, in DEGREES
for the angles, binned by the true value. Resolution is the std of the same
residual. Both come from predictions, so both are comparable across arms no
matter what objective produced them.

  CUDA_VISIBLE_DEVICES='' python make_bias_comparison.py
"""
import os, sys, json, gc
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
R = "/work/users/das214/SmartPixels/smart-pixels-ml"
sys.path.insert(0, os.path.join(R, "two_bit_optimization_helpers"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from prepare_tfrecords import load_tfrecords
from train import create_model
from conditional_nll import custom_loss_v2

# same BASE the driver and the perf-plot script use -- the dataset lives under
# /work/projects, not in the user's own /work/users tree
BASE = os.environ.get("SMARTPIX_BASE",
      "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_"
      "10ps_300k_convolved_to_200ps/shuffled_3d")
TFR = os.environ.get("SMARTPIX_TFR",
      os.path.join(BASE, "TFR_files_all101_noise_contained_discovery"))
OUT = os.environ.get("SMARTPIX_OUT", os.path.join(R, "runs", "perf_plots_o22")); os.makedirs(OUT, exist_ok=True)
MODEL = "ViT_MaxDeep_PairLattice"
LEVELS = np.array([0., 1., 2., 3.], np.float32)
NB = 15

# SMARTPIX_ARMS="label:runpath,label:runpath" overrides; default is the
# tolerance scan against the parent and the (already banked) control.
_default = [
    ("parent",   f"{R}/runs/o21v2a2_pairlattice/seed_22042/last.weights.hdf5", "#7c3aed"),
    ("control",  f"{R}/runs/o22_control_final/seed_22042/last.weights.hdf5",   "#64748b"),
    ("tol 0.50", f"{R}/runs/o22_tol050/seed_22042/last.weights.hdf5",          "#93c5fd"),
    ("tol 0.30", f"{R}/runs/o22_tol030/seed_22042/last.weights.hdf5",          "#3b82f6"),
    ("tol 0.17", f"{R}/runs/o22_tol017/seed_22042/last.weights.hdf5",          "#15803d"),
    ("tol 0.10", f"{R}/runs/o22_tol010/seed_22042/last.weights.hdf5",          "#b45309"),
]
_env = os.environ.get("SMARTPIX_ARMS", "")
if _env:
    # Generate one colour PER ARM. Zipping against a fixed palette silently
    # truncated the arm list to the palette length -- a 10-model comparison
    # evaluated only the first 6 and printed a table that looked complete.
    import matplotlib.cm as _cm
    _items = [a for a in _env.split(",") if a.strip()]
    _cols = ([ "#7c3aed" ] +
             [matplotlib.colors.to_hex(_cm.viridis(i / max(1, len(_items) - 2)))
              for i in range(len(_items) - 1)])
    ARMS = [(a.split(":", 1)[0], a.split(":", 1)[1], c) for a, c in zip(_items, _cols)]
    assert len(ARMS) == len(_items), "arm/colour length mismatch"
else:
    ARMS = _default

TARGETS = [("x", 0, "µm", False), ("y", 1, "µm", False),
           ("alpha", 2, "deg", True), ("beta", 3, "deg", True)]

meta = json.load(open(os.path.join(TFR, "TFR_test", "metadata.json")))
SCALE = np.asarray(meta["labels_scale"], dtype=np.float64)
_, vg = load_tfrecords(os.path.join(TFR, "TFR_train"),
                       os.path.join(TFR, "TFR_test"), noise=-1, seed=42)

def inv_cot(c):
    return np.arctan2(1.0, c) * 180.0 / np.pi

def evaluate(weights):
    tf.keras.backend.clear_session(); gc.collect()
    m = create_model(MODEL, timeslices=101, soft_quantize_layer=True,
                     initial_thresholds=[1., 2., 3.], threshold_offset=0.0,
                     initial_levels=LEVELS)
    m.load_weights(weights)
    idx = sorted(int(i) for i in np.array(
        m.get_layer('simple_router_output').selected_indices()).ravel())
    thr = [float(v) for v in np.array(m.get_layer('soft_quantizer_output').thresholds).ravel()]
    P, Y = [], []
    for i in range(len(vg)):
        X, y = vg[i]
        P.append(np.asarray(m.predict_on_batch(X))); Y.append(np.asarray(y))
    P = np.concatenate(P, 0); Y = np.concatenate(Y, 0)
    # Divide by the NUMBER OF BATCHES, not by events/5000. Keras reports
    # val_loss as the mean over batches of a batch-SUM loss, and the last batch
    # is short -- dividing by events/5000 gives a number ~5% off the ledger and
    # invites exactly the false comparison this column exists to prevent.
    nll = float(custom_loss_v2(tf.constant(Y, tf.float32),
                               tf.constant(P, tf.float32))) / max(1, len(vg))
    out = {"slices": idx, "thresholds": thr, "nll_val": nll, "n": len(Y)}
    for name, k, unit, is_ang in TARGETS:
        pred, true = P[:, 2 * k], Y[:, k]
        if is_ang:
            pred, true = inv_cot(pred * SCALE[k]), inv_cot(true * SCALE[k])
        else:
            pred, true = pred * SCALE[k], true * SCALE[k]
        res = true - pred
        lo, hi = np.percentile(true, [1, 99])
        edges = np.linspace(lo, hi, NB + 1)
        who = np.clip(np.digitize(true, edges) - 1, 0, NB - 1)
        mean = np.array([res[who == b].mean() if (who == b).sum() else np.nan for b in range(NB)])
        err  = np.array([res[who == b].std() / max(1, np.sqrt((who == b).sum()))
                         if (who == b).sum() else np.nan for b in range(NB)])
        out[name] = {"centres": (edges[:-1] + edges[1:]) / 2, "mean": mean, "err": err,
                     "sigma": float(res.std()), "unit": unit,
                     "max_abs_bias": float(np.nanmax(np.abs(mean))),
                     "worst_sigma_from_zero": float(np.nanmax(np.abs(mean / (err + 1e-12))))}
    del m
    return out

res = {}
for name, w, col in ARMS:
    if not os.path.exists(w):
        print(f"  {name}: {w} not present yet -- skipped"); continue
    print(f"evaluating {name} ...", flush=True)
    res[name] = evaluate(w); res[name]["color"] = col

if not res:
    raise SystemExit("nothing to compare yet")

# ---------------- figure ----------------
fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), dpi=115)
for ax, (name, k, unit, is_ang) in zip(axes.ravel(), TARGETS):
    for arm in [a for a, _, _ in ARMS if a in res]:
        if arm not in res: continue
        d = res[arm][name]
        ax.errorbar(d["centres"], d["mean"], yerr=3 * d["err"], lw=1.8, marker="o", ms=5,
                    color=res[arm]["color"], capsize=0,
                    label=f"{arm}  |bias| {d['max_abs_bias']:.2f}  σ {d['sigma']:.2f}")
    ax.axhline(0, ls="--", lw=1.2, c="#334155", zorder=1)
    ax.set_xlabel(f"true {name} [{unit}]", fontsize=11.5)
    ax.set_ylabel(f"true − predicted {name} [{unit}]", fontsize=11.5)
    ax.grid(alpha=.22, lw=.6); ax.legend(fontsize=9, framealpha=.94)
fig.suptitle("O22 tolerance scan — residual bias vs constraint strength",
             fontsize=14.5, y=.985)
fig.tight_layout(rect=[0, 0, 1, .97])
dst = os.path.join(OUT, "bias_comparison.png")
fig.savefig(dst, facecolor="white"); plt.close(fig)

# ---------------- table ----------------
print(f"\n{'target':<8}{'arm':<13}{'max|bias|':>11}{'worst nσ':>10}{'σ':>9}{'Δσ vs parent':>14}")
for name, k, unit, is_ang in TARGETS:
    for arm in [a for a, _, _ in ARMS if a in res]:
        if arm not in res: continue
        d = res[arm][name]; p = res["parent"][name] if "parent" in res else d
        dd = 100 * (d["sigma"] / p["sigma"] - 1)
        print(f"{name:<8}{arm:<13}{d['max_abs_bias']:>11.3f}{d['worst_sigma_from_zero']:>10.1f}"
              f"{d['sigma']:>9.3f}{dd:>13.1f}%")
print()
for arm in [a for a, _, _ in ARMS if a in res]:
    if arm in res:
        r = res[arm]
        print(f"{arm:<13} slices {r['slices']}  thr {[round(t,2) for t in r['thresholds']]}  "
              f"plain val NLL {r['nll_val']:,.0f}")
json.dump({a: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in ({kk: vv for kk, vv in d.items()} if isinstance(d, dict) else {}).items()}
           if isinstance(d, dict) else d
           for a, arm in res.items() for d in [arm]},
          open(os.path.join(OUT, "bias_comparison.json"), "w"), indent=1, default=str)
print("\nTRADE CURVE (bias reduction vs resolution cost, vs parent)")
print(f"{'arm':<12}" + "".join(f"{n+' bias':>13}" for n, _, _, _ in TARGETS)
      + "".join(f"{n+' dsig%':>13}" for n, _, _, _ in TARGETS))
for arm in [a for a, _, _ in ARMS if a in res]:
    row = f"{arm:<12}"
    for n, k, u, ang in TARGETS:
        row += f"{res[arm][n]['max_abs_bias']:>13.2f}"
    for n, k, u, ang in TARGETS:
        pv = res['parent'][n]['sigma'] if 'parent' in res else res[arm][n]['sigma']
        row += f"{100*(res[arm][n]['sigma']/pv-1):>+13.1f}"
    print(row)
print(f"\nwrote {dst}")
