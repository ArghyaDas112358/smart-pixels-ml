#!/bin/bash
# Part-1 LONG study on the interactive GPU (1000 epochs each, no early stopping,
# anneal stretched over all 1000, full per-epoch threshold/loss logging).
# Three batches, run sequentially:
#   A) 5 runs with RANDOM initial thresholds               -> runs/part1_long
#   B) 5 runs with FIXED initial thresholds [40,40.1,40.2]  -> runs/part1_long_fixed40
#   C) 5 runs with FIXED initial thresholds [20,20.1,20.2]  -> runs/part1_long_fixed20
# Resumable: skips any seed that already produced result.json; re-runs an
# interrupted one (its partial CSV is overwritten).
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
LOG=$REPO/runs/experiment_queue.log
cd "$REPO"

SEEDS="42 1042 2042 3042 4042"          # 5 runs per batch
RAND_OUT=$REPO/runs/part1_long
FIX_OUT=$REPO/runs/part1_long_fixed40
FIX20_OUT=$REPO/runs/part1_long_fixed20
FIXED="40,40.1,40.2"
FIXED20="20,20.1,20.2"

run_one () {  # $1=outdir  $2=seed  $3...=extra args
  local out=$1 seed=$2; shift 2
  mkdir -p "$out"
  if [ -f "$out/seed_$seed/result.json" ]; then
    echo "[$(date '+%H:%M:%S')] $(basename "$out") seed $seed already done, skipping" >> "$LOG"; return
  fi
  echo "[$(date '+%H:%M:%S')] LONG $(basename "$out") seed $seed (1000 ep) $*" >> "$LOG"
  $PY -u scripts/distillation/run_part1_long.py --seed "$seed" --epochs 1000 --out "$out" "$@" \
      >> "$out/seed_${seed}_console.log" 2>&1
  echo "[$(date '+%H:%M:%S')] $(basename "$out") seed $seed rc=$?" >> "$LOG"
}

echo "[$(date '+%H:%M:%S')] === BATCH A: 5 RANDOM-init runs ===" >> "$LOG"
for s in $SEEDS; do run_one "$RAND_OUT" "$s"; done

echo "[$(date '+%H:%M:%S')] === BATCH B: 5 FIXED-init [$FIXED] runs ===" >> "$LOG"
for s in $SEEDS; do run_one "$FIX_OUT" "$s" --init-thresholds "$FIXED"; done

echo "[$(date '+%H:%M:%S')] === BATCH C: 5 FIXED-init [$FIXED20] runs ===" >> "$LOG"
for s in $SEEDS; do run_one "$FIX20_OUT" "$s" --init-thresholds "$FIXED20"; done

echo "[$(date '+%H:%M:%S')] LONG study COMPLETE (5 random + 5 fixed40 + 5 fixed20)" >> "$LOG"
