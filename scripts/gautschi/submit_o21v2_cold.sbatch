#!/bin/bash -l
# NOTE the `-l`: Slurm batch scripts run in a NON-login shell where Lmod is not
# initialised and `module` does not exist.
# ---------------------------------------------------------------------------
# O21.v2a.2 (pair-lattice) and O21.v2c.2 (UCB bandit): the COLD, ZERO-PRIOR
# proof runs.
#
# The question: does ViT_d (deep decoder) under loss v2 find the optimal slice
# pair AND thresholds from scratch, with no inheritance from O17 or any earlier
# run? Everything is fresh:
#   * NO warm start -- backbone/head randomly initialised per seed
#   * router logits init to ZEROS = exactly uniform over all 5,050 pairs
#     (nothing hints at [11,26] or any other pair)
#   * thresholds drawn at random per seed, uniform(25,160), then trained
# If these land on the same pair/threshold family the warm-started runs found,
# the earlier results are proven free of warm-start bias.
#
# Ran here rather than on the AF because the AF session came back on a ~3 GiB
# GPU slice (was a full 40 GB A100) and cannot host a batch-5000 ViT_d worker.
#
# One arm+seed per job:  ARM=a SEED=22042 sbatch scripts/gautschi/submit_o21v2_cold.sbatch
#
# MEMORY: SMARTPIX_GPU_MEM_MB caps TF to 20 GB. Do NOT also set
# TF_GPU_ALLOCATOR=cuda_malloc_async -- it ignores the cap (see submit_o11).
# ---------------------------------------------------------------------------
#SBATCH --job-name=v2cold
#SBATCH -A cms
#SBATCH -p smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/v2cold_%x_%j.out
#SBATCH --error=logs/v2cold_%x_%j.err
set -euo pipefail
DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
SCRATCH=/scratch/gautschi/das214
REPO=$DEPOT/smart-pixels-ml
VENV=$SCRATCH/envs/smartpix-2bit
ARM=${ARM:?set ARM=a|c when submitting}
SEED=${SEED:?set SEED=<n> when submitting}
TARGET_EPOCHS=10000
case "$ARM" in
  a) MODEL=ViT_MaxDeep_PairLattice; OUT=$REPO/runs/o21v2a2_pairlattice ;;
  c) MODEL=ViT_MaxDeep_UCBRouter;   OUT=$REPO/runs/o21v2c2_ucb ;;
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
# v2 has no clip ceiling, and a COLD run's early val is legitimately enormous;
# the clip-era 9.9e4 threshold would execute healthy runs. NaN abort remains.
export SMARTPIX_ABORT_THR=1e12
# Kernel horizon scaled 1500 -> 3000 with the doubled budget: a cold head is
# still learning for the first ~1000 epochs, so the router must not commit
# while its reward signal is still noise. The bandit has no kernel.
if [[ "$ARM" == "a" ]]; then
  export SMARTPIX_SMOOTH_SIGMA0=4
  export SMARTPIX_SMOOTH_EPOCHS=3000
fi
# DELIBERATELY NOT SET: SMARTPIX_WARM_START / SMARTPIX_WARM_SRC_MODEL.
# That is the entire point of this campaign -- do not add them.
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-none} | O21.v2${ARM}.2 COLD seed $SEED model $MODEL ==="
echo "    warm start: NONE (cold) | loss: v2 | epochs: $TARGET_EPOCHS"
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
        echo "epoch $LAST < $((TARGET_EPOCHS-1)) -- resubmitting arm $ARM seed $SEED"
        cd "$REPO" && ARM=$ARM SEED=$SEED sbatch scripts/gautschi/submit_o21v2_cold.sbatch
    else
        echo "arm $ARM seed $SEED reached epoch $LAST -- done"
    fi
fi
