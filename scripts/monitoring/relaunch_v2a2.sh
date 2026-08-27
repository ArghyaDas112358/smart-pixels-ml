#!/bin/bash
# Sequential relaunch for the O21.v2a.2 / v2c.2 cold runs.
#
# The first attempt started four workers 40 s apart and three OOM'd: peak GPU
# demand is during graph build + first step, so overlapping four of those on
# one 40 GB A100 collides even though the STEADY state of four fits. Fix: add
# one worker at a time and do not start the next until the previous has
# written its first epoch row (i.e. is past the allocation peak).
R=/work/users/das214/SmartPixels/smart-pixels-ml
SC=/tmp/claude-978920/-work-users-das214-SmartPixels/7a0a041e-b87d-47e5-ae22-8b5c100f4193/scratchpad
PY=/work/users/das214/envs/smartpix-2bit/bin/python
EPOCHS=10000

wait_training () {    # wait_training <run_dir> <seed> <max_wait_s>
  local d=$1/seed_$2 waited=0
  while [ $waited -lt $3 ]; do
    if [ -f "$d/history.csv" ] && [ "$(wc -l < "$d/history.csv")" -ge 2 ]; then
      echo "   -> seed $2 training (epoch row present after ${waited}s)"; return 0
    fi
    sleep 20; waited=$((waited+20))
  done
  echo "   !! seed $2 no epoch row after ${waited}s"; return 1
}

launch () {           # launch <arm> <seed>
  local arm=$1 seed=$2 model outdir smooth=()
  case "$arm" in
    a) model=ViT_MaxDeep_PairLattice; outdir=$R/runs/o21v2a2_pairlattice
       smooth=(SMARTPIX_SMOOTH_SIGMA0=4 SMARTPIX_SMOOTH_EPOCHS=3000) ;;
    c) model=ViT_MaxDeep_UCBRouter;   outdir=$R/runs/o21v2c2_ucb ;;
  esac
  # a previous OOM leaves a seed_<n> dir that scan_existing reads as "attempted"
  # and would silently skip the relaunch -- move it out of the seed_* namespace.
  [ -d "$outdir/seed_$seed" ] && mv "$outdir/seed_$seed" "$outdir/oom_${seed}_$(date +%H%M%S)"
  cd "$R" && env "${smooth[@]}" \
    SMARTPIX_MODEL_NAME=$model SMARTPIX_LOSS_V2=1 SMARTPIX_ABORT_THR=1e12 \
    nohup $PY scripts/distillation/run_simplerouter_mdmm_discovery.py \
      --epochs $EPOCHS --seeds "$seed" --target 1 --out "$outdir" \
      > "$SC/v2${arm}2_${seed}.log" 2>&1 &
  echo "$(date +%H:%M) launched O21.v2${arm}.2 seed $seed pid $!"
  wait_training "$outdir" "$seed" 600
}

launch a 22042
launch a 22142
launch c 22142
echo "AF workers: $(ps -eo comm,args | awk '$1 ~ /^python/ && /run_simplerouter_mdmm_discovery/' | wc -l)"
