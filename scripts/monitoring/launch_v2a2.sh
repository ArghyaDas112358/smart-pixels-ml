#!/bin/bash
# O21.v2a.2 / O21.v2c.2 -- the COLD, ZERO-PRIOR proof runs.
#
# Everything fresh: NO warm start (so no backbone/head/threshold inheritance
# from O17), router logits start at zeros = exactly uniform over all 5,050
# pairs, thresholds drawn at random per seed from uniform(25,160). Loss v2
# (softplus diag, unclipped) + ViT_d (deep decoder). The claim under test:
# this model+loss finds the optimal slice pair AND thresholds unaided.
#
# Two mechanisms, same seeds, so the arms are paired:
#   O21.v2a.2 = pair-lattice router   (ViT_MaxDeep_PairLattice, 2D pair kernel)
#   O21.v2c.2 = UCB bandit router     (ViT_MaxDeep_UCBRouter, no kernel)
#
# AF cap is 4 workers (a 5th OOMs), so this starts seeds 22042/22142 of each
# arm here; seed 22242 of each arm runs on Gautschi as a resubmitting chain.
R=/work/users/das214/SmartPixels/smart-pixels-ml
SC=/tmp/claude-978920/-work-users-das214-SmartPixels/7a0a041e-b87d-47e5-ae22-8b5c100f4193/scratchpad
PY=/work/users/das214/envs/smartpix-2bit/bin/python
EPOCHS=10000

launch () {           # launch <arm> <seed>
  local arm=$1 seed=$2 model outdir
  case "$arm" in
    a) model=ViT_MaxDeep_PairLattice; outdir=$R/runs/o21v2a2_pairlattice ;;
    c) model=ViT_MaxDeep_UCBRouter;   outdir=$R/runs/o21v2c2_ucb ;;
  esac
  # sigma horizon scaled 1500 -> 3000 with the doubled budget: a COLD head is
  # still learning for the first ~1000 epochs, so the router must not commit
  # while its reward signal is still noise. UCB has no kernel (SMOOTH unset).
  local smooth=()
  [ "$arm" = a ] && smooth=(SMARTPIX_SMOOTH_SIGMA0=4 SMARTPIX_SMOOTH_EPOCHS=3000)
  cd "$R" && env "${smooth[@]}" \
    SMARTPIX_MODEL_NAME=$model \
    SMARTPIX_LOSS_V2=1 \
    SMARTPIX_ABORT_THR=1e12 \
    nohup $PY scripts/distillation/run_simplerouter_mdmm_discovery.py \
      --epochs $EPOCHS --seeds "$seed" --target 1 --out "$outdir" \
      > "$SC/v2${arm}2_${seed}.log" 2>&1 &
  echo "$(date +%H:%M) launched O21.v2${arm}.2 seed $seed ($model) pid $!"
  sleep 40
}

launch a 22042
launch a 22142
launch c 22042
launch c 22142
echo "AF workers now: $(ps -eo comm,args | awk '$1 ~ /^python/ && /run_simplerouter_mdmm_discovery/' | wc -l)"
