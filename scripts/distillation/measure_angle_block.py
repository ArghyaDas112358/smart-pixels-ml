"""
Measure the conditional angle cost -log p(cotA, cotB | x, y) on the FULL
validation set, per finished seed. This is what the new MDMM constraint bounds,
so the target is read off real runs instead of invented the way corr >= 0.5 was.

  CUDA_VISIBLE_DEVICES='' python measure_angle_block.py 2042 4042 3042 1042
"""
import os, sys, gc, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
R = "/work/users/das214/SmartPixels/smart-pixels-ml"
sys.path.insert(0, os.path.join(R, "two_bit_optimization_helpers"))
import tensorflow as tf
from conditional_nll import nll_terms
from prepare_tfrecords import load_tfrecords
from train import create_model

RUN = os.path.join(R, "runs", os.environ.get("SMARTPIX_RUN_DIR", "simplerouter_mdmm_discovery"))
MODEL = os.environ.get("SMARTPIX_MODEL", "ViT_Max_SimpleRouter")
BASE = ("/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence"
        "_10ps_300k_convolved_to_200ps/shuffled_3d")
TFR = os.path.join(BASE, "TFR_files_all101_noise_contained_discovery")
SEEDS = [int(s) for s in sys.argv[1:]] or [2042, 4042, 3042, 1042]

_, vg = load_tfrecords(os.path.join(TFR, "TFR_train"), os.path.join(TFR, "TFR_test"),
                       noise=-1, seed=42, shuffle=False)

print(f"{'seed':>5} {'slices':>10} {'x':>8} {'y|x':>8} {'cotA|xy':>9} {'cotB|..':>9} "
      f"{'POS':>9} {'ANGLE':>9} {'joint':>9}")
print("-" * 84)
rows = {}
for s in SEEDS:
    tf.keras.backend.clear_session(); gc.collect()
    m = create_model(MODEL, timeslices=101, soft_quantize_layer=True,
                     initial_thresholds=[1., 2., 3.], threshold_offset=0.0,
                     initial_levels=np.array([0., 1., 2., 3.], dtype=np.float32))
    m.load_weights(os.path.join(RUN, f"seed_{s}", "last.weights.hdf5"))
    acc, n = np.zeros(4), 0
    for i in range(len(vg)):
        X, y = vg[i]
        t = nll_terms(tf.cast(y, tf.float64), tf.cast(m(X, training=False), tf.float64)).numpy()
        acc += t.sum(0); n += len(t)
    t = acc / n
    rj = os.path.join(RUN, f"seed_{s}", "result.json")
    sl = json.load(open(rj))["final_indices"] if os.path.exists(rj) else "-"
    rows[s] = dict(terms=t.tolist(), pos=float(t[0]+t[1]), ang=float(t[2]+t[3]),
                   joint=float(t.sum()), slices=sl, events=n)
    print(f"{s:>5} {str(sl):>10} {t[0]:>8.4f} {t[1]:>8.4f} {t[2]:>9.4f} {t[3]:>9.4f} "
          f"{t[0]+t[1]:>9.4f} {t[2]+t[3]:>9.4f} {t.sum():>9.4f}")

best = min(rows.items(), key=lambda kv: kv[1]["ang"])
print(f"\nbest angle block: seed {best[0]} at {best[1]['ang']:.4f} "
      f"(slices {best[1]['slices']}), over {best[1]['events']:,} events")
json.dump(rows, open(os.path.join(R, "runs", "angle_block_measured.json"), "w"), indent=1)
print("wrote runs/angle_block_measured.json")
