#!/bin/bash
# Run the 1000-epoch distillation sweep + final report, sequentially.
# Designed to be launched detached. Reads teacher ckpt from
# runs/vit_max_run_1000ep/summary.json (written by extract_teacher_from_part1.py).
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
SWEEP_DIR=$REPO/runs/distill_sweep_vit_teacher_1000ep
TEACHER_DIR=$REPO/runs/vit_max_run_1000ep
LOG=$SWEEP_DIR/queue.log
mkdir -p "$SWEEP_DIR"
{
  echo "[$(date '+%H:%M:%S')] launcher pid=$$"
  CKPT=$($PY -c "import json; print(json.load(open('$TEACHER_DIR/summary.json'))['part2_checkpoints'])")
  echo "[$(date '+%H:%M:%S')] teacher ckpt=$CKPT"
  cd "$REPO"
  $PY -u scripts/distillation/distill_sweep.py \
    --teacher-checkpoints "$CKPT" \
    --teacher-model-type ViT_Max \
    --thresholds-json "$TEACHER_DIR/optimized_thresholds.json" \
    --epochs 1000 --warmup-steps 200 \
    --out "$SWEEP_DIR"
  RC=$?
  echo "[$(date '+%H:%M:%S')] sweep rc=$RC"
  if [[ $RC -eq 0 ]]; then
    $PY scripts/distillation/save_student_parquets.py \
      --sweep-dir "$SWEEP_DIR" \
      --teacher-summary "$TEACHER_DIR/summary.json" \
      --thresholds-json "$TEACHER_DIR/optimized_thresholds.json" \
      --out-dir "$REPO/runs/processed_parquets/test_3src/2bit_optimized_students_1000ep"
    $PY scripts/distillation/report_final_1000ep.py
  fi
} >> "$LOG" 2>&1
