"""
Is Pearson correlation the right thing to constrain?

Pearson is invariant to scale, so a model that predicts a SHRUNK version of the
target -- pred = a*true + b with a << 1, the classic regress-to-the-mean
degenerate fit -- keeps a high correlation while throwing away most of the
dynamic range. Our summary plots show exactly that signature (a strong slope in
"true - pred" vs "true" for both angles), so the question is whether
MinCorrConstraint is blind to the pathology it was hired to prevent.

For each seed this prints, per target:
    corr        Pearson (what MDMM constrains)
    slope       OLS slope of pred on true = corr * sd(pred)/sd(true)
                1.0 = no shrinkage; < 1 = predictions compressed toward the mean
    sd ratio    sd(pred)/sd(true)

A constraint on `slope` is not gameable the way corr or sd alone are: inflating
the prediction variance without tracking the target leaves corr ~ 0, so the
product stays near 0.

  CUDA_VISIBLE_DEVICES='' python check_shrinkage.py 2042 4042 3042
"""
import os, sys, json, gc
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
sys.path.insert(0, os.path.join(R, "two_bit_optimization_helpers"))
import tensorflow as tf
from prepare_tfrecords import load_tfrecords
from train import create_model

RUN = os.path.join(R, "runs", os.environ.get("SMARTPIX_RUN_DIR", "simplerouter_mdmm_discovery"))
MODEL = os.environ.get("SMARTPIX_MODEL", "ViT_Max_SimpleRouter")
BASE = ("/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence"
        "_10ps_300k_convolved_to_200ps/shuffled_3d")
TFR = os.path.join(BASE, "TFR_files_all101_noise_contained_discovery")
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)
SEEDS = [int(s) for s in sys.argv[1:]] or [2042, 4042]

_, vg = load_tfrecords(os.path.join(TFR, "TFR_train"), os.path.join(TFR, "TFR_test"),
                       noise=-1, seed=42, shuffle=False)

# model output is the 14-vector with means interleaved at 0,2,4,6; labels are the
# normalized 4-vector (same layout MDMM_OUTPUT_COLUMNS / MDMM_LABEL_COLUMNS use).
OUT_COL = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
LAB_COL = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}


def predict(seed):
    tf.keras.backend.clear_session(); gc.collect()
    m = create_model(MODEL, timeslices=101, soft_quantize_layer=True,
                     initial_thresholds=[1., 2., 3.], threshold_offset=0.0,
                     initial_levels=LEVELS)
    m.load_weights(os.path.join(RUN, f"seed_{seed}", "last.weights.hdf5"))
    P, T = [], []
    for i in range(len(vg)):
        X, y = vg[i]
        P.append(np.asarray(m(X, training=False))); T.append(np.asarray(y))
    return np.concatenate(P), np.concatenate(T)


print(f"{'seed':>5} {'target':>6} {'corr':>7} {'slope':>7} {'sd ratio':>9}   {'verdict':<28}")
print("-" * 74)
for s in SEEDS:
    P, T = predict(s)
    for k in ("x", "y", "cotA", "cotB"):
        p = P[:, OUT_COL[k]].astype(np.float64)
        t = T[:, LAB_COL[k]].astype(np.float64)
        corr = float(np.corrcoef(p, t)[0, 1])
        sdr = float(p.std() / t.std())
        slope = corr * sdr                     # OLS slope of pred on true
        note = ("ok" if slope > 0.9 else
                f"shrunk to {slope*100:.0f}% of true range")
        print(f"{s:>5} {k:>6} {corr:>7.3f} {slope:>7.3f} {sdr:>9.3f}   {note:<28}")
    print()
