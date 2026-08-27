#!/bin/bash -l
# NOTE the `-l`: Slurm batch scripts run in a NON-login shell where Lmod is not
# initialised and `module` does not exist.
# ---------------------------------------------------------------------------
# O11 on Gautschi: the PLAIN MDMM x SimpleRouter seeds (no beta annealing).
#
# Split of work (user, 2026-08-01): the AF A100 runs the priority O4/beta
# campaign; the confirmatory O11 seeds live here, where queue latency is
# harmless because their value is statistical rather than urgent.
#
# One seed per job:   SEED=6042 sbatch scripts/gautschi/submit_o11.sbatch
#
# MEMORY: SMARTPIX_GPU_MEM_MB caps TF to a fixed 20 GB pool. Do NOT also set
# TF_GPU_ALLOCATOR=cuda_malloc_async -- the async allocator IGNORES that cap, and
# with both set the pool ate the whole 47.7 GB card (O4 jobs died at epochs 311,
# 563 and 817, the last one with CUDA reporting 99 MB free and allocator name
# "gpu_async_0"). BFC + a hard cap is the combination that respects the limit.
# This is UNTESTED at length -- the fix was never exercised before O4 moved to
# AF -- so watch the first few hundred epochs.
#
# Self-chaining across the 12h cap, safe because resume is exact: last.weights
# every epoch + optimizer state (warm Nadam moments) + Lagrange multipliers, all
# in the run directory.
# ---------------------------------------------------------------------------
#SBATCH --job-name=o11router
#SBATCH -A cms
#SBATCH -p smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/o11_%x_%j_%a.out
#SBATCH --error=logs/o11_%x_%j_%a.err

set -euo pipefail

DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
SCRATCH=/scratch/gautschi/das214
REPO=$DEPOT/smart-pixels-ml
VENV=$SCRATCH/envs/smartpix-2bit

SEED=${SEED:?set SEED=<n> when submitting}
TARGET_EPOCHS=5000
OUT=$REPO/runs/simplerouter_mdmm_discovery

mkdir -p logs "$OUT"

module --ignore_cache load modtree/cpu
module --ignore_cache load python/3.11.9
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export SMARTPIX_TFR=$SCRATCH/TFR_files_all101_noise_contained_discovery
export TF_CPP_MIN_LOG_LEVEL=1
export SMARTPIX_GPU_MEM_MB=20000

echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-none} | O11 seed $SEED ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
# `|| true`: this GPU leaks ~170 MB/epoch (proven: ~253 epochs before OOM with the
# whole 47 GB card, ~123 with a 20 GB cap -- half the memory, half the epochs), so
# a chunk is EXPECTED to die of OOM partway. Without this, `set -e` aborts the
# script at the failure and the resubmit block below never runs, which is why the
# first four O11 jobs stopped dead instead of chaining. Resume is exact, so each
# chunk still advances ~120 epochs and the chain grinds forward unattended while
# the leak is investigated. The same code does NOT leak on the AF A100.
python scripts/distillation/run_simplerouter_mdmm_discovery.py \
    --epochs "$TARGET_EPOCHS" --seeds "$SEED" --extend --target 1 --out "$OUT" || true

# ---- resubmit if this seed has not reached the target ----------------------
HIST=$OUT/seed_$SEED/history.csv
if [[ -f $HIST ]]; then
    LAST=$(tail -1 "$HIST" | cut -d, -f1 | tr -d '\r')
    if [[ "$LAST" =~ ^[0-9]+$ ]] && (( LAST < TARGET_EPOCHS - 1 )); then
        echo "epoch $LAST < $((TARGET_EPOCHS-1)) -- resubmitting seed $SEED"
        cd "$REPO" && SEED=$SEED sbatch scripts/gautschi/submit_o11.sbatch
    else
        echo "seed $SEED reached epoch $LAST -- done"
    fi
fi
