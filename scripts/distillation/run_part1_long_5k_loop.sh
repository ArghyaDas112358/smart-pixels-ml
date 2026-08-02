#!/bin/bash
# Part-1 LONG study v2 — 5000 epochs, EarlyStopping(patience=500, restore_best_weights),
# anneal k 1->67 stretched over the full 5000 epochs, per-epoch threshold/loss logging.
# Three batches, 5 seeds each, run sequentially on the interactive GPU:
#   A) RANDOM init                  -> runs/part1_long_5k
#   B) FIXED [40,40.1,40.2]         -> runs/part1_long_5k_fixed40
#   C) FIXED [20,20.1,20.2]         -> runs/part1_long_5k_fixed20
# Resumable: skips any seed with result.json; re-runs an interrupted one.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
LOG=$REPO/runs/experiment_queue.log
cd "$REPO"

SEEDS="42 1042 2042 3042 4042"
EPOCHS=5000
PATIENCE=500
RAND_OUT=$REPO/runs/part1_long_5k
FIX40_OUT=$REPO/runs/part1_long_5k_fixed40
FIX20_OUT=$REPO/runs/part1_long_5k_fixed20

run_one () {  # $1=outdir  $2=seed  $3...=extra args
  local out=$1 seed=$2; shift 2
  mkdir -p "$out"
  if [ -f "$out/seed_$seed/result.json" ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] $(basename "$out") seed $seed already done, skipping" >> "$LOG"; return
  fi
  echo "[$(date '+%m-%d %H:%M:%S')] LONG5k $(basename "$out") seed $seed (max $EPOCHS ep, patience $PATIENCE) $*" >> "$LOG"
  $PY -u scripts/distillation/run_part1_long.py --seed "$seed" --epochs $EPOCHS --patience $PATIENCE --out "$out" "$@" \
      >> "$out/seed_${seed}_console.log" 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] $(basename "$out") seed $seed rc=$?" >> "$LOG"
}

echo "[$(date '+%m-%d %H:%M:%S')] === v2 BATCH A: 5 RANDOM-init runs (5000ep/p500) ===" >> "$LOG"
for s in $SEEDS; do run_one "$RAND_OUT" "$s"; done

echo "[$(date '+%m-%d %H:%M:%S')] === v2 BATCH B: 5 FIXED-init [40,40.1,40.2] runs ===" >> "$LOG"
for s in $SEEDS; do run_one "$FIX40_OUT" "$s" --init-thresholds "40,40.1,40.2"; done

echo "[$(date '+%m-%d %H:%M:%S')] === v2 BATCH C: 5 FIXED-init [20,20.1,20.2] runs ===" >> "$LOG"
for s in $SEEDS; do run_one "$FIX20_OUT" "$s" --init-thresholds "20,20.1,20.2"; done

echo "[$(date '+%m-%d %H:%M:%S')] LONG5k study COMPLETE (5 random + 5 fixed40 + 5 fixed20)" >> "$LOG"
