#!/bin/bash -l
# NOTE the `-l`: Slurm batch scripts run in a NON-login shell where Lmod is not
# initialised and `module` does not exist.
# ---------------------------------------------------------------------------
# O12 on Gautschi: MDMM x SimpleRouter with the CONDITIONAL-NLL constraint.
#
# Moved here from the AF (user, 2026-08-04) so the A100 can run O13, which
# replaces O12's single constraint on the angle BLOCK with two multipliers, one
# per angle. O12's seeds resume EXACTLY from their AF checkpoints: last.weights
# every epoch + optimizer moments + the Lagrange multiplier.
#
# One seed per job:  SEED=12142 sbatch scripts/gautschi/submit_o12.sbatch
#
# The constraint is  -log p(cotA,cotB | x,y) <= -2.58  (per-event nats, measured
# frontier is -2.2764 on O11 seed 2042). LOSS_CLIP=1 keeps the task loss
# bit-identical to O11/O4 so the constraint is the only difference.
#
# MEMORY: SMARTPIX_GPU_MEM_MB caps TF to 20 GB. Do NOT also set
# TF_GPU_ALLOCATOR=cuda_malloc_async -- it ignores the cap (see submit_o11).
# ---------------------------------------------------------------------------
#SBATCH --job-name=o12router
#SBATCH -A cms
#SBATCH -p smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/o12_%x_%j.out
#SBATCH --error=logs/o12_%x_%j.err
set -euo pipefail
DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
SCRATCH=/scratch/gautschi/das214
REPO=$DEPOT/smart-pixels-ml
VENV=$SCRATCH/envs/smartpix-2bit
SEED=${SEED:?set SEED=<n> when submitting}
TARGET_EPOCHS=5000
OUT=$REPO/runs/simplerouter_mdmm_o12nll
mkdir -p logs "$OUT"
module --ignore_cache load modtree/cpu
module --ignore_cache load python/3.11.9
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export SMARTPIX_TFR=$SCRATCH/TFR_files_all101_noise_contained_discovery
export TF_CPP_MIN_LOG_LEVEL=1
export SMARTPIX_GPU_MEM_MB=20000
# --- the O12 constraint (defaults in the driver are the O11 behaviour) ---
export SMARTPIX_CONSTRAINT=anglenll
export SMARTPIX_NLL_BLOCK=angle
export SMARTPIX_ANGLE_NLL_TARGET=-2.58
export SMARTPIX_LOSS_CLIP=1
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-none} | O12 seed $SEED ==="
echo "    constraint: -log p($SMARTPIX_NLL_BLOCK) <= $SMARTPIX_ANGLE_NLL_TARGET, clip=$SMARTPIX_LOSS_CLIP"
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
        cd "$REPO" && SEED=$SEED sbatch scripts/gautschi/submit_o12.sbatch
    else
        echo "seed $SEED reached epoch $LAST -- done"
    fi
fi
