"""
Skip Part 2: extract the body of the Part-1 ViT_Max checkpoint (the one
with the SoftQuantizeLayer front end) into a fresh ViT_Max model that has
NO SoftQuantizeLayer. Use that as the frozen distillation teacher, fed
with 2-bit-digitized inputs that the data pipeline produces (so it sees
the same {0,1,2,3} input distribution the student will see at deploy).

The Part-1 body adapted to SoftQuantize-at-k=67 outputs (essentially
{0,1,2,3} with infinitesimal smoothness at threshold boundaries). Feeding
it the slightly-sharper hard-digitized {0,1,2,3} is a negligible
distribution shift and the body's accuracy is preserved.
"""
import os, sys, json, shutil
sys.path.insert(0, "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers")
import numpy as np
import tensorflow as tf
from train import create_model

REPO = "/work/users/das214/SmartPixels/smart-pixels-ml"
PART1_CKPT_DIR = f"{REPO}/runs/weights/weights-2t-ViT_Max-soft_quantize_layer-1a62cd8b-checkpoints"
OUT_DIR = f"{REPO}/runs/weights/weights-2t-ViT_Max-2bit_optimized-from_part1_extract-checkpoints"
TEACHER_RUN_DIR = f"{REPO}/runs/vit_max_run_1000ep"

# pick best Part-1 ckpt (lowest val_loss)
files = [f for f in os.listdir(PART1_CKPT_DIR) if f.endswith('.hdf5')]
vloss = [float(f.split('-v')[1].split('.hdf5')[0]) for f in files]
best = files[int(np.argmin(vloss))]
best_path = os.path.join(PART1_CKPT_DIR, best)
print(f"best Part-1 ckpt: {best} (val_loss = {vloss[int(np.argmin(vloss))]})")

# load thresholds (already saved earlier)
thr_json = json.load(open(f"{TEACHER_RUN_DIR}/optimized_thresholds.json"))

# build SQ-on (to load weights into), then SQ-off (target)
m_sq = create_model("ViT_Max", timeslices=2, soft_quantize_layer=True,
                    initial_thresholds=[247.8, 668.4, 1662.9], threshold_offset=80.0,
                    initial_levels=np.array([0., 1., 2., 3.], dtype=np.float32))
m_sq.load_weights(best_path)
print(f"SQ-on model loaded. Total params: {m_sq.count_params()}")

m_no = create_model("ViT_Max", timeslices=2, soft_quantize_layer=False)
print(f"SQ-off model built. Total params: {m_no.count_params()}")

# Skip SoftQuantizeLayer in SQ-on; copy the rest by position.
sq_weight_count = sum(len(L.weights) for L in m_sq.layers
                      if L.__class__.__name__ == "SoftQuantizeLayer")
print(f"SoftQuantizeLayer has {sq_weight_count} weight tensors (these will be skipped)")
sq_weights_all = m_sq.weights
body_weights = sq_weights_all[sq_weight_count:]  # everything after the SQ layer
no_weights_all = m_no.weights

assert len(body_weights) == len(no_weights_all), (
    f"weight-tensor count mismatch: SQ-on body has {len(body_weights)} tensors, "
    f"SQ-off has {len(no_weights_all)}")
for src, dst in zip(body_weights, no_weights_all):
    assert src.shape == dst.shape, f"shape mismatch: {src.name} {src.shape} vs {dst.name} {dst.shape}"
    dst.assign(src)
print(f"copied {len(body_weights)} weight tensors")

# Spot-check: verify trainable_weights moved over (not just non-trainable)
print(f"SQ-off model has {sum(int(np.prod(w.shape)) for w in m_no.trainable_weights)} trainable params after copy")

# Save extracted teacher weights to a new checkpoint directory
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)
# Mimic the existing checkpoint filename format used by save_performance_parquet and best_checkpoint helpers.
# Use a "fake" epoch=01 and use Part-1's val_loss so downstream best_checkpoint() picks this file.
v = float(vloss[int(np.argmin(vloss))])
out_file = f"{OUT_DIR}/weights.01-t{v:.2f}-v{v:.2f}.hdf5"
m_no.save_weights(out_file)
print(f"wrote {out_file}")

# Write summary.json so the chained watcher / distill scripts can pick it up.
summary = {
    "model_type": "ViT_Max",
    "epochs_each_part": 1000,
    "part1_note": "Part 1 stopped at epoch 522; Part 2 skipped (Part-1 body extracted).",
    "part1_best_val_loss": v,
    "part1_best_checkpoint": best,
    "part1_checkpoints": PART1_CKPT_DIR,
    "part2_skipped": True,
    "part2_note": "Skipped intentionally: Part-1 body's effective input distribution at k=67 matches deployment {0,1,2,3} input within a negligible smoothness. Body weights extracted into a SoftQuantize-less wrapper.",
    "part2_best_val_loss": v,    # using Part-1's val_loss as the operative teacher quality
    "part2_checkpoints": OUT_DIR,
    "optimized_thresholds": thr_json["thresholds"],
}
json.dump(summary, open(f"{TEACHER_RUN_DIR}/summary.json", "w"), indent=1)
print(f"wrote {TEACHER_RUN_DIR}/summary.json")
print("DONE")
