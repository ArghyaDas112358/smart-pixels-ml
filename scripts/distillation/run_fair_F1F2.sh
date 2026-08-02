#!/bin/bash
# Runs F1 (TBR) then F2 (DeiT) in the FAIR regime, AFTER the solo F0 (w=0
# validation) run frees the GPU. Waits for F0 to write summary.json (success)
# or FAILED.txt before starting. All three use fit_with_retry (Nadam, no
# clipnorm, abort-on-stuck, auto-reseed) so the comparison is apples-to-apples.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
LOG=$REPO/runs/experiment_queue.log
cd "$REPO"

F0_DONE="runs/distill_qmlp_moe_w0_fair/summary.json"
F0_FAIL="runs/distill_qmlp_moe_w0_fair/FAILED.txt"

echo "[$(date '+%H:%M:%S')] F1F2 queue waiting for F0 to free the GPU" >> "$LOG"
waited=0
while [ ! -f "$F0_DONE" ] && [ ! -f "$F0_FAIL" ]; do
  sleep 60
  waited=$((waited+1))
  if [ "$waited" -gt 600 ]; then
    echo "[$(date '+%H:%M:%S')] F1F2 queue gave up waiting for F0 (10h)" >> "$LOG"
    exit 1
  fi
done
echo "[$(date '+%H:%M:%S')] F0 finished; starting F1" >> "$LOG"

run() {
  local label="$1"; shift
  echo "[$(date '+%H:%M:%S')] START $label" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "[$(date '+%H:%M:%S')] END   $label (rc=$?)" >> "$LOG"
}

run "F1_tbr_fair_w1" \
  $PY -u scripts/distillation/train_distill_tbr.py \
    --student-model-type QMlp_MoE_Max --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --w 1.0 \
    --out runs/distill_qmlp_moe_tbr_fair_w1

run "F2_deit_fair_w0p5" \
  $PY -u scripts/distillation/train_distill_deit.py \
    --student-model-type QMlp_MoE_Max_Distill --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --w 0.5 \
    --out runs/distill_qmlp_moe_deit_fair_w0p5

echo "[$(date '+%H:%M:%S')] F1F2 queue COMPLETE" >> "$LOG"
