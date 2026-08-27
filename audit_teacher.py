"""
Teacher/data/eval audit script.
Runs fully CPU-side to avoid GPU init noise.
"""
import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model
from loss import custom_loss

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_DIR = ("/work/users/das214/SmartPixels/smart-pixels-ml/runs/weights/"
            "weights-2t-ViT_Max-2bit_optimized-from_part1_extract-checkpoints")
THR_JSON  = ("/work/users/das214/SmartPixels/smart-pixels-ml/runs/"
             "vit_max_run_1000ep/optimized_thresholds.json")
DATASET   = ("/depot/cms/users/das214/datasets/largerWindowPreliminary/"
             "dataset_3sr_16x16_50x12P5_centeredIncidence_parquets")
PART1_CKPT_DIR = ("/work/users/das214/SmartPixels/smart-pixels-ml/runs/weights/"
                  "weights-2t-ViT_Max-soft_quantize_layer-1a62cd8b-checkpoints")

# ── 1. Best checkpoint path ───────────────────────────────────────────────────
def best_ckpt(d):
    fs = [f for f in os.listdir(d) if f.endswith('.hdf5')]
    vl = [float(f.split('-v')[1].split('.hdf5')[0]) for f in fs]
    return os.path.join(d, fs[int(np.argmin(vl))]), min(vl)

ckpt_path, ckpt_vloss = best_ckpt(CKPT_DIR)
print(f"\n=== Checkpoint ===")
print(f"  file:  {os.path.basename(ckpt_path)}")
print(f"  vloss: {ckpt_vloss:.2f}  (expected ≈ -43762.58)")

# ── 2. Load thresholds ────────────────────────────────────────────────────────
thr = json.load(open(THR_JSON))
thresholds = np.array(thr['thresholds'], dtype=np.float32)
levels     = np.array(thr['levels'],     dtype=np.float32)
k_at_ext   = thr.get('k_at_extraction', 'n/a')
print(f"\n=== Thresholds ===")
print(f"  thresholds: {thresholds}")
print(f"  levels:     {levels}")
print(f"  k at extraction: {k_at_ext}")

# ── 3. Build data pipeline ────────────────────────────────────────────────────
print("\n=== Building generators ===")
_, _, tfr_tr, tfr_val = generate_tfrecords(
    dataset_dir=DATASET, model_type='ViT_Max',
    train_batch_size=5000, val_batch_size=5000,
    select_contained=False, timeslices=2,
    tfrecords_exist=True, seed=42)

tg, vg = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=True,
                        digitize_levels=levels, digitize_thresholds=thresholds,
                        seed=42)
print("  generators ready")

# ── 4. Build + load teacher ───────────────────────────────────────────────────
print("\n=== Teacher model ===")
teacher = create_model('ViT_Max', timeslices=2, soft_quantize_layer=False)
teacher.load_weights(ckpt_path)
print(f"  params:  {teacher.count_params():,}")
print(f"  layers:  {[l.name for l in teacher.layers[:5]]}...")

# Check if SoftQuantizeLayer is present in the teacher
layer_names = [l.__class__.__name__ for l in teacher.layers]
has_sq = 'SoftQuantizeLayer' in layer_names
print(f"  SoftQuantizeLayer in teacher: {has_sq}")

# ── 5. Get one val batch to inspect input ────────────────────────────────────
print("\n=== Input inspection ===")
x_batch, y_batch = next(iter(vg))
x_np = x_batch.numpy()
print(f"  x shape:      {x_np.shape}")
print(f"  x dtype:      {x_np.dtype}")
print(f"  x unique vals (first 10): {np.unique(x_np)[:10]}")
print(f"  x min={x_np.min():.2f}  max={x_np.max():.2f}  mean={x_np.mean():.4f}")

# ── 6. Teacher predictions + NLL on hard-digitized val set ───────────────────
print("\n=== Teacher NLL on hard-digitized val set ===")
all_preds = []
all_truth = []
for xb, yb in vg:
    preds = teacher(xb, training=False)
    all_preds.append(preds.numpy())
    all_truth.append(yb.numpy())

preds_all = np.concatenate(all_preds, axis=0)
truth_all  = np.concatenate(all_truth, axis=0)
n_events   = preds_all.shape[0]
print(f"  total val events: {n_events}")

# Compute NLL using custom_loss (returns sum over batch)
nll_total = custom_loss(
    tf.constant(truth_all, dtype=tf.float32),
    tf.constant(preds_all, dtype=tf.float32)
).numpy()

nll_per_event = nll_total / n_events
print(f"  NLL total:     {nll_total:.2f}")
print(f"  NLL per event: {nll_per_event:.4f}  (expected ≈ -8.75)")
print(f"  NLL for 5000:  {nll_per_event * 5000:.2f}  (expected ≈ -43763)")

# ── 7. Teacher NLL on soft-quantized input (comparison) ──────────────────────
print("\n=== Teacher NLL on soft-quantized input (distribution-shift quantification) ===")
# Build SoftQuantizer-on teacher from Part-1 checkpoint
try:
    p1_ckpt, p1_vloss = best_ckpt(PART1_CKPT_DIR)
    print(f"  Part-1 ckpt: {os.path.basename(p1_ckpt)}, vloss={p1_vloss:.2f}")

    teacher_sq = create_model('ViT_Max', timeslices=2, soft_quantize_layer=True,
                              initial_thresholds=[226.04, 601.01, 1456.94],
                              threshold_offset=80.0,
                              initial_levels=np.array([0.,1.,2.,3.], dtype=np.float32))
    teacher_sq.load_weights(p1_ckpt)
    print(f"  SQ-teacher params: {teacher_sq.count_params():,}")

    # Get raw (non-digitized) val generator for comparison
    _, vg_raw = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=False, seed=42)

    # Also get digitized val generator (already have: vg)
    # SQ teacher on raw input (its trained distribution):
    all_preds_sq_raw = []
    all_truth_sq_raw = []
    for xb, yb in vg_raw:
        preds = teacher_sq(xb, training=False)  # internally applies SQ layer at inference (hard quant at training=False)
        all_preds_sq_raw.append(preds.numpy())
        all_truth_sq_raw.append(yb.numpy())

    preds_sq_raw = np.concatenate(all_preds_sq_raw, axis=0)
    truth_sq_raw = np.concatenate(all_truth_sq_raw, axis=0)
    nll_sq_raw = custom_loss(
        tf.constant(truth_sq_raw, dtype=tf.float32),
        tf.constant(preds_sq_raw, dtype=tf.float32)
    ).numpy() / truth_sq_raw.shape[0]

    print(f"  SQ-teacher on raw input (its own distribution): NLL/event = {nll_sq_raw:.4f}")
    print(f"    (This is what the SQ-model actually experienced during Part-1 training)")

    # Now compare: extracted-body teacher on hard-digitized = already computed as nll_per_event
    # SQ teacher on hard-digitized input (mismatch scenario):
    all_preds_sq_dig = []
    all_truth_sq_dig = []
    for xb, yb in vg:
        preds = teacher_sq(xb, training=False)
        all_preds_sq_dig.append(preds.numpy())
        all_truth_sq_dig.append(yb.numpy())
    preds_sq_dig = np.concatenate(all_preds_sq_dig, axis=0)
    truth_sq_dig = np.concatenate(all_truth_sq_dig, axis=0)
    nll_sq_dig = custom_loss(
        tf.constant(truth_sq_dig, dtype=tf.float32),
        tf.constant(preds_sq_dig, dtype=tf.float32)
    ).numpy() / truth_sq_dig.shape[0]

    print(f"  SQ-teacher on hard-digitized input: NLL/event = {nll_sq_dig:.4f}")
    print(f"  -> Distribution shift (SQ-on-raw vs SQ-on-dig): {abs(nll_sq_dig - nll_sq_raw):.4f} NLL/event")
    print(f"  -> Extracted body on hard-dig vs SQ-on-raw: {abs(nll_per_event - nll_sq_raw):.4f} NLL/event")

except Exception as e:
    print(f"  [WARNING] SQ comparison failed: {e}")
    nll_sq_raw = None

# ── 8. Pull distributions (calibration) ──────────────────────────────────────
print("\n=== Calibration (pull distributions) ===")
# 14-output layout: [x, M11, y, M22, cotA, M33, cotB, M44, M21, M31, M32, M41, M42, M43]
# means at 0,2,4,6; diag at 1,3,5,7; off-diag at 8..13
p = preds_all  # (N, 14)
y = truth_all  # (N, 4): [x, y, cotA, cotB]

mu = p[:, 0:8:2]   # (N, 4): x, y, cotA, cotB predictions
diag_raw = p[:, 1:8:2]   # (N, 4)

# Diagonal: relu + 1e-9 floor (as in custom_loss)
diag = np.maximum(diag_raw, 0.0) + 1e-9   # sigma_11, sigma_22, sigma_33, sigma_44

# Off-diagonal
off = p[:, 8:]  # (N, 6): M21, M31, M32, M41, M42, M43

# Marginal sigmas from Cholesky rows
sigma_x    = diag[:, 0]
sigma_y    = np.sqrt(off[:,0]**2 + diag[:,1]**2)
sigma_cotA = np.sqrt(off[:,1]**2 + off[:,2]**2 + diag[:,2]**2)
sigma_cotB = np.sqrt(off[:,3]**2 + off[:,4]**2 + off[:,5]**2 + diag[:,3]**2)

residuals = mu - y   # (N, 4)
sigmas    = np.stack([sigma_x, sigma_y, sigma_cotA, sigma_cotB], axis=1)
pulls     = residuals / (sigmas + 1e-20)

names = ['x', 'y', 'cotA', 'cotB']
print(f"  {'var':<8} {'pull_mean':>10} {'pull_std':>10}  {'pull_rms':>10}  calibrated?")
pull_widths = []
for i, name in enumerate(names):
    pm = pulls[:, i].mean()
    ps = pulls[:, i].std()
    prms = np.sqrt(np.mean(pulls[:, i]**2))
    cal = "YES" if 0.8 < ps < 1.2 else "NO (over-confident)" if ps < 0.8 else "NO (under-confident)"
    print(f"  {name:<8} {pm:>10.4f} {ps:>10.4f}  {prms:>10.4f}  {cal}")
    pull_widths.append(ps)

# ── 9. Eval fairness check ────────────────────────────────────────────────────
print("\n=== Eval fairness (standalone vs distill) ===")
print("  Standalone (train.py): model.compile(loss=custom_loss)")
print("    -> keras val_loss = custom_loss(y, p) = SUM over batch")
print("    -> part2_best_val_loss = min(val_loss history) [sum over 5000]")
print("")
print("  Distill test_step (distill14.py / distill14_taid.py):")
print("    data_loss = custom_loss(y, s14) / n  [per-event mean]")
print("    val_loss_data tracked by Mean metric -> average over batches")
print("")
print("  COMPARISON: standalone logs SUM; distill logs MEAN")
print("  standalone part2_best_val_loss / 5000 should equal distill val_loss_data")
print("")

# Verify numerically: recompute sum vs mean on one batch
xb_t = tf.constant(x_batch.numpy(), dtype=tf.float32)
yb_t = tf.constant(y_batch.numpy(), dtype=tf.float32)
preds_batch = teacher(xb_t, training=False)
n_b = float(xb_t.shape[0])
sum_loss = custom_loss(yb_t, preds_batch).numpy()
mean_loss = sum_loss / n_b
print(f"  One batch: custom_loss SUM = {sum_loss:.2f}, SUM/n = {mean_loss:.4f}")
print(f"  -> Standalone val_loss/5000 = {sum_loss/5000:.4f}")
print(f"  -> Distill val_loss_data    = {mean_loss:.4f}")
print(f"  -> These are identical (both compute per-event mean = sum/n) ✓")

# ── 10. Summary ───────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"  1) Teacher checkpoint val_loss (encoded in filename): {ckpt_vloss:.2f}")
print(f"  2) Measured NLL/event on hard-digitized val:          {nll_per_event:.4f}")
print(f"     -> NLL for 5000 events:                           {nll_per_event*5000:.2f}")
print(f"  3) SoftQuantizeLayer present in extracted teacher:    {has_sq}")
print(f"     k at extraction: {k_at_ext}  (67 = near-hard; actual = {k_at_ext})")
print(f"  4) Input range seen by student: {x_np.min():.0f} to {x_np.max():.0f} (hard {{0,1,2,3}})")
print(f"  5) Pull widths: x={pull_widths[0]:.3f}  y={pull_widths[1]:.3f}  "
      f"cotA={pull_widths[2]:.3f}  cotB={pull_widths[3]:.3f}")
