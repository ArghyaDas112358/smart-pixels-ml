#!/bin/bash
# Part-1 (ViT + SoftQuantize threshold optimization) on the new 3srb dataset,
# with retry-on-OOM: the shared GPU is sometimes too full to train the ViT at
# batch 5000 (user spec: keep 5000, change nothing). If the run dies with
# ResourceExhausted/OOM, wait 15 min for the GPU to free up and retry the
# IDENTICAL run. Stops on success (optimized_thresholds.json) or a non-OOM
# failure.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
OUT=$REPO/runs/new_dataset_part1
LOG=$REPO/runs/experiment_queue.log
cd "$REPO"

for a in $(seq 1 24); do
  [ -f "$OUT/optimized_thresholds.json" ] && { echo "[$(date '+%H:%M:%S')] Part-1 DONE (thresholds extracted)" >> "$LOG"; break; }
  rm -f "$OUT/FAILED.txt"
  echo "[$(date '+%H:%M:%S')] Part-1 attempt $a (batch 5000, samples [11,26])" >> "$LOG"
  $PY -u scripts/distillation/run_part1_new_dataset.py >> "$OUT/console.log" 2>&1
  if [ -f "$OUT/optimized_thresholds.json" ]; then
    echo "[$(date '+%H:%M:%S')] Part-1 SUCCEEDED on attempt $a" >> "$LOG"; break
  fi
  if [ -f "$OUT/FAILED.txt" ] && grep -qaE "ResourceExhausted|OOM" "$OUT/FAILED.txt"; then
    echo "[$(date '+%H:%M:%S')] Part-1 attempt $a OOM; GPU busy -- retrying in 15 min" >> "$LOG"
    sleep 900
  else
    echo "[$(date '+%H:%M:%S')] Part-1 attempt $a non-OOM failure -- stopping for review" >> "$LOG"
    break
  fi
done
