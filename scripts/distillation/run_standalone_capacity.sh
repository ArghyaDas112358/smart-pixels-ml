#!/bin/bash
# STANDALONE capacity curve (pure NLL, NO teacher) at M/L/XL. The foundational
# experiment: how far does just adding params get us toward the teacher
# (-8.75/event), with no distillation? Robust (no forward-KL explosion). Uses
# the validated w=0 recipe (train_distill_tbr.py --w 0 == F0, which reproduced
# the -30,816 standalone at tiny size). Fair pipeline (Nadam, retry-on-stuck).
#
# Tells us: (a) the capacity-vs-NLL curve, (b) whether capacity alone reaches
# the teacher, (c) the warm-start weights for a later gentle-KL refinement.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
LOG=$REPO/runs/experiment_queue.log
cd "$REPO"

run() {
  local label="$1"; shift
  echo "[$(date '+%H:%M:%S')] START $label" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "[$(date '+%H:%M:%S')] END   $label (rc=$?)" >> "$LOG"
}

echo "[$(date '+%H:%M:%S')] standalone capacity sweep start" >> "$LOG"

run "S_M_w0_11k" \
  $PY -u scripts/distillation/train_distill_tbr.py \
    --student-model-type QMlp_MoE_M --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --w 0.0 \
    --out runs/standalone_moe_M

run "S_L_w0_31k" \
  $PY -u scripts/distillation/train_distill_tbr.py \
    --student-model-type QMlp_MoE_L --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --w 0.0 \
    --out runs/standalone_moe_L

run "S_XL_w0_85k" \
  $PY -u scripts/distillation/train_distill_tbr.py \
    --student-model-type QMlp_MoE_XL --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --w 0.0 \
    --out runs/standalone_moe_XL

echo "[$(date '+%H:%M:%S')] standalone capacity sweep COMPLETE" >> "$LOG"
