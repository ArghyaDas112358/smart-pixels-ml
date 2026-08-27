#!/bin/bash
# Resilient launcher for the 8-bit QAT small chain. The trainer checkpoints
# every epoch and auto-resumes from runs/.../ckpt.weights.h5, so re-running this
# script after a GPU preemption continues from the last saved epoch. Does NOT
# delete the out dir (that would wipe the checkpoint). Loops until summary.json
# (converged) or FAILED.txt.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
OUT=$REPO/runs/slimmable_chain_small_q8
LOG=$REPO/runs/experiment_queue.log
cd "$REPO"
mkdir -p "$OUT"

for a in $(seq 1 50); do
  [ -f "$OUT/summary.json" ] && { echo "[$(date '+%H:%M:%S')] q8 chain CONVERGED" >> "$LOG"; break; }
  [ -f "$OUT/FAILED.txt" ] && { echo "[$(date '+%H:%M:%S')] q8 chain FAILED" >> "$LOG"; break; }
  echo "[$(date '+%H:%M:%S')] q8 chain attempt $a (resumes from ckpt if present)" >> "$LOG"
  $PY -u scripts/distillation/train_slimmable.py \
    --widths-preset small --quantize \
    --attach-vit --lam-vit 0.5 --vit-tau 1.5 --vit-warmup 200 --lam-kd 1.0 --clipnorm 1.0 \
    --epochs 1000 --patience 50 \
    --out "$OUT" >> "$OUT/console.log" 2>&1
  sleep 3
done
