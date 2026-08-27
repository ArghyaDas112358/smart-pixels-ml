#!/bin/bash -l
# NOTE the `-l`: Slurm batch scripts run in a NON-login shell where Lmod is not
# initialised and `module` does not exist.
# ---------------------------------------------------------------------------
# O21.v2 on Gautschi: the O21 mechanism arms RETRAINED under LOSS V2
# (softplus diagonal, unclipped log-domain NLL -- conditional_nll.custom_loss_v2,
# tfp-free). Same warm start (O17 escapee 41942), same seeds as O21 so the
# comparison is paired; ONLY the loss dialect changes. v2 NLLs are NOT
# comparable with the clip-dialect O11..O21 numbers (~7 nats/event of pure
# dialect artifact was measured cross-scoring; never mix in one table).
# Expect an early recalibration dip: the warm checkpoint's diag outputs are
# relu-dialect and the head re-fits them under softplus in the first epochs.
#
# One arm per job:  ARM=a sbatch scripts/gautschi/submit_o21v2.sbatch
#   a = pair-lattice (2D pair-space kernel)   seed 21042
#   b = two independent routers               seed 21142
#   c = discounted-UCB bandit                 seed 21242
#
# MEMORY: SMARTPIX_GPU_MEM_MB caps TF to 20 GB. Do NOT also set
# TF_GPU_ALLOCATOR=cuda_malloc_async -- it ignores the cap (see submit_o11).
# ---------------------------------------------------------------------------
#SBATCH --job-name=o21v2
#SBATCH -A cms
#SBATCH -p smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/o21v2_%x_%j.out
#SBATCH --error=logs/o21v2_%x_%j.err
set -euo pipefail
DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
SCRATCH=/scratch/gautschi/das214
REPO=$DEPOT/smart-pixels-ml
VENV=$SCRATCH/envs/smartpix-2bit
ARM=${ARM:?set ARM=a|b|c when submitting}
TARGET_EPOCHS=5000
case "$ARM" in
  a) SEED=21042; MODEL=ViT_MaxDeep_PairLattice;     SMOOTH=1; OUT=$REPO/runs/o21v2a_pairlattice ;;
  b) SEED=21142; MODEL=ViT_MaxDeep_TwoRouterSmooth; SMOOTH=1; OUT=$REPO/runs/o21v2b_tworouter ;;
  c) SEED=21242; MODEL=ViT_MaxDeep_UCBRouter;       SMOOTH=0; OUT=$REPO/runs/o21v2c_ucb ;;
  *) echo "bad ARM=$ARM"; exit 1 ;;
esac
mkdir -p logs "$OUT"
module --ignore_cache load modtree/cpu
module --ignore_cache load python/3.11.9
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export SMARTPIX_TFR=$SCRATCH/TFR_files_all101_noise_contained_discovery
export TF_CPP_MIN_LOG_LEVEL=1
export SMARTPIX_GPU_MEM_MB=20000
export SMARTPIX_MODEL_NAME=$MODEL
export SMARTPIX_LOSS_V2=1
# v2 val has no clip ceiling: the bandit's forced sweep legitimately spikes
# val past the clip-dialect 9.9e4 plateau threshold, which struck out arm c's
# first chunks. Effectively disable the threshold test; NaN abort remains.
export SMARTPIX_ABORT_THR=1e12
export SMARTPIX_WARM_START=$REPO/ckpt/o17_41942_best.weights.hdf5
export SMARTPIX_WARM_SRC_MODEL=ViT_MaxDeep_SimpleRouter
if [[ "$SMOOTH" == "1" ]]; then
  export SMARTPIX_SMOOTH_SIGMA0=4
  export SMARTPIX_SMOOTH_EPOCHS=1500
fi
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-none} | O21.v2 arm $ARM seed $SEED model $MODEL ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$REPO"
# `|| true`: this GPU leaks ~170 MB/epoch, so a chunk is EXPECTED to die of OOM
# partway. Resume is exact, so the chain grinds forward regardless.
python scripts/distillation/run_simplerouter_mdmm_discovery.py \
    --epochs "$TARGET_EPOCHS" --seeds "$SEED" --extend --target 1 --out "$OUT" || true
# ---- resubmit if this arm has not reached the target -----------------------
HIST=$OUT/seed_$SEED/history.csv
if [[ -f $HIST ]]; then
    LAST=$(tail -1 "$HIST" | cut -d, -f1 | tr -d '\r')
    if [[ "$LAST" =~ ^[0-9]+$ ]] && (( LAST < TARGET_EPOCHS - 1 )); then
        echo "epoch $LAST < $((TARGET_EPOCHS-1)) -- resubmitting arm $ARM"
        # RESUBMIT GUARD: sbatch can be REFUSED (QOS job cap). The old line
        # ignored its exit status, so the chain ended silently -- seeds 22042
        # and 24142 each sat dead for hours that way. Retry, then shout.
        for _try in 1 2 3 4 5; do
            if cd "$REPO" && ARM=$ARM sbatch scripts/gautschi/submit_o21v2.sbatch; then
                echo "resubmit ok (attempt $_try)"; break
            fi
            echo "RESUBMIT FAILED (attempt $_try) -- retrying in 120s"
            if [ "$_try" = 5 ]; then echo "RESUBMIT GUARD: GIVING UP -- CHAIN DEAD"; else sleep 120; fi
        done
    else
        echo "arm $ARM reached epoch $LAST -- done"
    fi
fi
