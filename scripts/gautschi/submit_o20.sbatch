#!/bin/bash -l
# NOTE the `-l`: Slurm batch scripts run in a NON-login shell where Lmod is not
# initialised and `module` does not exist.
# ---------------------------------------------------------------------------
# O20 on Gautschi: the WARM-START PROBE.
#
# O18 (deep head + kernel + free router, cold start) reproduced O15's late
# window -- capacity alone does not flip the free router. O20 asks the
# chicken-and-egg question directly: start from O17 escapee 41942's weights
# (backbone + head + thresholds DEVELOPED at [11,26], best -38,994) but with
# the ROUTER at its fresh flat init, then run the exact O18 recipe. If the
# developed representation flips the router early, the landscape story is
# proven; if the head re-adapts late anyway, the late attractor beats even a
# formed early representation.
#
# The driver applies SMARTPIX_WARM_START on FRESH starts only, so this
# self-resubmitting chain resumes chunks from its own last.weights as usual.
#
# One seed per job:  SEED=20042 sbatch scripts/gautschi/submit_o20.sbatch
#
# MEMORY: SMARTPIX_GPU_MEM_MB caps TF to 20 GB. Do NOT also set
# TF_GPU_ALLOCATOR=cuda_malloc_async -- it ignores the cap (see submit_o11).
# ---------------------------------------------------------------------------
#SBATCH --job-name=o20warm
#SBATCH -A cms
#SBATCH -p smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/o20_%x_%j.out
#SBATCH --error=logs/o20_%x_%j.err
set -euo pipefail
DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
SCRATCH=/scratch/gautschi/das214
REPO=$DEPOT/smart-pixels-ml
VENV=$SCRATCH/envs/smartpix-2bit
SEED=${SEED:?set SEED=<n> when submitting}
TARGET_EPOCHS=5000
OUT=$REPO/runs/simplerouter_o20warm
mkdir -p logs "$OUT"
module --ignore_cache load modtree/cpu
module --ignore_cache load python/3.11.9
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export SMARTPIX_TFR=$SCRATCH/TFR_files_all101_noise_contained_discovery
export TF_CPP_MIN_LOG_LEVEL=1
export SMARTPIX_GPU_MEM_MB=20000
# --- the O18 recipe (defaults = O11 corr constraints), plus the warm start ---
export SMARTPIX_MODEL_NAME=ViT_MaxDeep_SimpleRouterSmooth
export SMARTPIX_SMOOTH_SIGMA0=4
export SMARTPIX_SMOOTH_EPOCHS=1500
export SMARTPIX_WARM_START=$REPO/ckpt/o17_41942_best.weights.hdf5
export SMARTPIX_WARM_SRC_MODEL=ViT_MaxDeep_SimpleRouter
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-none} | O20 seed $SEED ==="
echo "    warm start: $SMARTPIX_WARM_START (router theta fresh)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$REPO"
# `|| true`: this GPU leaks ~170 MB/epoch, so a chunk is EXPECTED to die of OOM
# partway. Resume is exact, so the chain grinds forward regardless.
python scripts/distillation/run_simplerouter_mdmm_discovery.py \
    --epochs "$TARGET_EPOCHS" --seeds "$SEED" --extend --target 1 --out "$OUT" || true
# ---- resubmit if this seed has not reached the target ----------------------
HIST=$OUT/seed_$SEED/history.csv
if [[ -f $HIST ]]; then
    LAST=$(tail -1 "$HIST" | cut -d, -f1 | tr -d '\r')
    if [[ "$LAST" =~ ^[0-9]+$ ]] && (( LAST < TARGET_EPOCHS - 1 )); then
        echo "epoch $LAST < $((TARGET_EPOCHS-1)) -- resubmitting seed $SEED"
        # RESUBMIT GUARD: sbatch can be REFUSED (QOS job cap). The old line
        # ignored its exit status, so the chain ended silently -- seeds 22042
        # and 24142 each sat dead for hours that way. Retry, then shout.
        for _try in 1 2 3 4 5; do
            if cd "$REPO" && SEED=$SEED sbatch scripts/gautschi/submit_o20.sbatch; then
                echo "resubmit ok (attempt $_try)"; break
            fi
            echo "RESUBMIT FAILED (attempt $_try) -- retrying in 120s"
            if [ "$_try" = 5 ]; then echo "RESUBMIT GUARD: GIVING UP -- CHAIN DEAD"; else sleep 120; fi
        done
    else
        echo "seed $SEED reached epoch $LAST -- done"
    fi
fi
