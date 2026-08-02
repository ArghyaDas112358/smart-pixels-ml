#!/bin/bash
# Resilient driver for the Part-1 threshold COLLECTION. The python script loops
# over seeds, appends each successful run's thresholds to threshold_runs.jsonl,
# and writes median_thresholds.json when it finishes (target reached / seeds
# exhausted). It is resumable (skips already-collected seeds), so if a run is
# preempted this wrapper just re-runs it and the collection continues.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
OUT=$REPO/runs/new_dataset_part1
LOG=$REPO/runs/experiment_queue.log
cd "$REPO"
for a in $(seq 1 100); do
  [ -f "$OUT/median_thresholds.json" ] && { echo "[$(date '+%H:%M:%S')] collection COMPLETE" >> "$LOG"; break; }
  echo "[$(date '+%H:%M:%S')] Part-1 collection runner attempt $a (resumes JSONL)" >> "$LOG"
  rm -f "$OUT/FAILED.txt"
  $PY -u scripts/distillation/run_part1_new_dataset.py >> "$OUT/console.log" 2>&1
  sleep 5
done
