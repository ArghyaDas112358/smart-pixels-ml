#!/bin/bash
# FAIR queue. The earlier distillation runs were confounded: they used the
# distill driver (Adam + clipnorm=1.0, EarlyStopping patience=50, stops ~ep128)
# while the standalone -30,526 baseline used Nadam, no clipnorm, and ran to
# ep420 (best). So every distillation run was judged at <1/3 of the training the
# baseline got. These runs use the SAME regime as the standalone (Nadam, no
# clipnorm) so w=0 should reproduce ~-30,526 and TBR/DeiT become honest tests.
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

echo "[$(date '+%H:%M:%S')] FAIR queue start" >> "$LOG"

# F0 -- pure NLL control in the FAIR pipeline. Must reproduce ~-30,526 to prove
# the pipeline is fair. If it does, the old 'distillation loses' result was a
# training-setup artifact, not a property of distillation.
run "F0_w0_fair_nadam" \
  $PY -u scripts/distillation/train_distill_tbr.py \
    --student-model-type QMlp_MoE_Max --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --w 0.0 --optimizer nadam --clipnorm 0 \
    --out runs/distill_qmlp_moe_w0_fair

# F1 -- TBR (teacher-bounded mean hinge) in the fair pipeline.
run "F1_tbr_fair_w1" \
  $PY -u scripts/distillation/train_distill_tbr.py \
    --student-model-type QMlp_MoE_Max --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --w 1.0 --optimizer nadam --clipnorm 0 \
    --out runs/distill_qmlp_moe_tbr_fair_w1

# F2 -- DeiT separate-head in the fair pipeline.
run "F2_deit_fair_w0p5" \
  $PY -u scripts/distillation/train_distill_deit.py \
    --student-model-type QMlp_MoE_Max_Distill --teacher-model-type ViT_Max \
    --epochs 1000 --patience 50 --w 0.5 --optimizer nadam --clipnorm 0 \
    --out runs/distill_qmlp_moe_deit_fair_w0p5

echo "[$(date '+%H:%M:%S')] FAIR queue COMPLETE" >> "$LOG"
