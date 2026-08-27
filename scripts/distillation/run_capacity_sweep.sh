#!/bin/bash
# CAPACITY-SCALING SWEEP toward the teacher (-8.75/event = -43,762/batch).
# Goal: find the SMALLEST MoE student that, distilled from the ViT_Max teacher
# with full forward-KL in the fair pipeline, reaches the teacher.
#
# Ladder: M(11K) -> L(31K) -> XL(85K)  vs  teacher 419K.
# Recipe: data-NLL + forward-KL[T||S], fixed beta=1.0, warmup 200, Nadam,
# abort-on-stuck + auto-reseed (fair_fit). Smallest first so the curve emerges.
# Waits for the solo F0 (w=0 tiny baseline) to free the GPU first.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
LOG=$REPO/runs/experiment_queue.log
cd "$REPO"

F0_DONE="runs/distill_qmlp_moe_w0_fair/summary.json"
F0_FAIL="runs/distill_qmlp_moe_w0_fair/FAILED.txt"
echo "[$(date '+%H:%M:%S')] capacity sweep waiting for F0 to free the GPU" >> "$LOG"
waited=0
while [ ! -f "$F0_DONE" ] && [ ! -f "$F0_FAIL" ]; do
  sleep 60; waited=$((waited+1))
  if [ "$waited" -gt 600 ]; then
    echo "[$(date '+%H:%M:%S')] capacity sweep gave up waiting for F0 (10h)" >> "$LOG"; exit 1
  fi
done
echo "[$(date '+%H:%M:%S')] F0 done; starting capacity sweep" >> "$LOG"

run() {
  local label="$1"; shift
  echo "[$(date '+%H:%M:%S')] START $label" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "[$(date '+%H:%M:%S')] END   $label (rc=$?)" >> "$LOG"
}

run "C_M_kl11k" \
  $PY -u scripts/distillation/train_distill_kl_fair.py \
    --student-model-type QMlp_MoE_M --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --fixed-beta 1.0 --warmup-steps 200 \
    --out runs/distill_moe_M_klfair

run "C_L_kl31k" \
  $PY -u scripts/distillation/train_distill_kl_fair.py \
    --student-model-type QMlp_MoE_L --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --fixed-beta 1.0 --warmup-steps 200 \
    --out runs/distill_moe_L_klfair

run "C_XL_kl85k" \
  $PY -u scripts/distillation/train_distill_kl_fair.py \
    --student-model-type QMlp_MoE_XL --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --fixed-beta 1.0 --warmup-steps 200 \
    --out runs/distill_moe_XL_klfair

echo "[$(date '+%H:%M:%S')] capacity sweep COMPLETE" >> "$LOG"
