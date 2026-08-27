#!/bin/bash
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
TA_DIR=$REPO/runs/distill_conv2d_max_from_vit
TA_WEIGHTS=$TA_DIR/student_final.weights.h5
OUT=$REPO/runs/distill_qconv2d_takd_hint
LOG=$OUT/chain.log
mkdir -p "$OUT"
{
  echo "[$(date '+%H:%M:%S')] waiting for TA (Conv2D_Max distilled from ViT) to finish"
  until [ -f "$TA_DIR/summary.json" ]; do
    if ! pgrep -f "train_distill_qconv2d.py.*Conv2D_Max" >/dev/null; then
      [ -f "$TA_DIR/summary.json" ] || { echo "ABORT: TA process gone, no summary"; exit 1; }
    fi
    sleep 180
  done
  TA_NLL=$($PY -c "import json;print(json.load(open('$TA_DIR/summary.json'))['best_val_loss_data'])")
  echo "[$(date '+%H:%M:%S')] TA done. best_val_loss_data=$TA_NLL/event. Launching TAKD+hint."
  cd "$REPO"
  $PY -u scripts/distillation/train_distill_hint.py \
    --student-model-type QConv2D_Max \
    --teacher-model-type Conv2D_Max \
    --teacher-weights-file "$TA_WEIGHTS" \
    --epochs 1000 --patience 50 --beta-hint 1.0 \
    --out "$OUT" >> "$OUT/train_outer.log" 2>&1
  echo "[$(date '+%H:%M:%S')] TAKD+hint finished rc=$?"
  [ -f "$OUT/summary.json" ] && cat "$OUT/summary.json"
} >> "$LOG" 2>&1
