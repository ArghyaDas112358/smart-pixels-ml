#!/bin/bash
# CORRECTED experiment queue (post grad/KL audit). Floor fixed to 1e-4 in
# distill.py; gradient norms now balanced (forward-KL ~= data at beta=1.0 on a
# trained student). Recipe: FORWARD KL (audit-confirmed correct), tiny-enough
# fixed beta, warmup so KL engages after the student has a baseline covariance.
# Each experiment must clear standalone -30,526/batch (teacher -44,622).
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
LOG=$REPO/runs/experiment_queue.log
cd "$REPO"

run_and_wait() {
  local label="$1"; shift
  echo "[$(date '+%H:%M:%S')] START $label" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "[$(date '+%H:%M:%S')] END   $label (rc=$?)" >> "$LOG"
}

echo "[$(date '+%H:%M:%S')] CORRECTED queue start (GPU free)" >> "$LOG"

# E1 — Forward-KD, FIXED+FLOORED+WARMUP, beta=1.0 (gradient-equalized). The
# cleanest test that the two confirmed bugs (floor, MDMM runaway) were the cause.
run_and_wait "fwdKD_b1.0_warm1500" \
  $PY -u scripts/distillation/train_distill_qconv2d.py \
    --student-model-type QMlp_MoE_Max --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --fixed-beta 1.0 --warmup-steps 1500 \
    --out runs/distill_qmlp_moe_fwd_b1

# E1b — lighter beta=0.3 (data dominates more)
run_and_wait "fwdKD_b0.3_warm1500" \
  $PY -u scripts/distillation/train_distill_qconv2d.py \
    --student-model-type QMlp_MoE_Max --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --fixed-beta 0.3 --warmup-steps 1500 \
    --out runs/distill_qmlp_moe_fwd_b0p3

# E1c — heavier beta=3.0 (teacher pulls more)
run_and_wait "fwdKD_b3.0_warm1500" \
  $PY -u scripts/distillation/train_distill_qconv2d.py \
    --student-model-type QMlp_MoE_Max --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --fixed-beta 3.0 --warmup-steps 1500 \
    --out runs/distill_qmlp_moe_fwd_b3

# E3 — TAID (distribution-space interp + reverse KL to nearby target), floored
run_and_wait "taid_distspace_b1.0" \
  $PY -u scripts/distillation/train_distill_taid.py \
    --student-model-type QMlp_MoE_Max --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --beta 1.0 --anneal-steps 20000 \
    --out runs/distill_qmlp_moe_taid_v2

echo "[$(date '+%H:%M:%S')] QUEUE COMPLETE" >> "$LOG"
