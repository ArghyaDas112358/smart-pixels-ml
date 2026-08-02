#!/bin/bash
# ---------------------------------------------------------------------------
# O4 on Gautschi: MDMM x SimpleRouter with an ANNEALED pair-distribution
# temperature (beta 1 -> 30), which is the fix for the router never committing
# (measured on AF: mu spread over ~60 effective slices after 2500 epochs, so one
# threshold triple had to serve amplitudes from 74 to 335 mV).
#
# EDIT BEFORE FIRST USE:
#   -A   your allocation      (list them with: slist)
#   -p   the GPU partition    (list them with: sinfo -s   -- likely 'ai')
#
# Submit:   cd $DEPOT/smart-pixels-ml && sbatch scripts/gautschi/submit_o4.sbatch
# Status:   squeue --me
#
# SELF-CHAINING: Slurm caps walltime below the length of a 5000-epoch run, so the
# job resubmits itself until the seed reaches the target epoch. This is safe
# because the driver resumes exactly (last.weights.hdf5 every epoch + Lagrange
# multipliers replayed from mdmm_epochs.csv + a FIXED 1000-epoch anneal horizon
# so a resumed run never re-softens an already-annealed quantizer).
# CAVEAT: resume does NOT restore the Nadam optimizer moments. With a 24 h chunk
# and ~30 s/epoch on H100 that is only ~1-2 resumes over the whole run, which is
# tolerable; if you drop to the 4 h standby QoS it becomes ~10 resets and is
# worth implementing optimizer checkpointing first.
# ---------------------------------------------------------------------------
#SBATCH --job-name=o4router
#SBATCH -A cms                      # <-- EDIT: your allocation (slist)
#SBATCH -p ai                       # <-- EDIT: GPU partition (sinfo -s)
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/o4_%x_%j.out
#SBATCH --error=logs/o4_%x_%j.err

set -euo pipefail

DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
REPO=$DEPOT/smart-pixels-ml
ENV_PREFIX=$DEPOT/envs/smartpix-2bit

SEED=${SEED:-8042}                  # a seed NOT used by the AF O11 campaign
TARGET_EPOCHS=5000
BETA_FINAL=30                       # top-2 mu mass ~1.66 at beta=20, ~1.97 at 60
BETA_START=1000                     # hold beta=1 until the quantizer anneal ends
BETA_EPOCHS=2000                    # then ramp over 2000 epochs, then hold
OUT=$REPO/runs/simplerouter_mdmm_o4beta          # SEPARATE dir: beta checkpoints
                                                 # are incompatible with plain ones

mkdir -p logs "$OUT"

module purge
module load conda || module load anaconda
# shellcheck disable=SC1091
source activate "$ENV_PREFIX" 2>/dev/null || conda activate "$ENV_PREFIX"

# point the driver at the Depot-staged dataset instead of the AF /work paths
export SMARTPIX_TFR=$DEPOT/TFR_files_all101_noise_contained_discovery
export TF_CPP_MIN_LOG_LEVEL=1

echo "=== $(date) | host $(hostname) | job $SLURM_JOB_ID | seed $SEED ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
python scripts/distillation/run_simplerouter_mdmm_discovery.py \
    --epochs "$TARGET_EPOCHS" --seeds "$SEED" --extend --target 1 \
    --beta-final "$BETA_FINAL" --beta-start "$BETA_START" --beta-epochs "$BETA_EPOCHS" \
    --out "$OUT"

# ---- resubmit if the seed has not reached the target -----------------------
HIST=$OUT/seed_$SEED/history.csv
if [[ -f $HIST ]]; then
    LAST=$(tail -1 "$HIST" | cut -d, -f1 | tr -d '\r')
    if [[ "$LAST" =~ ^[0-9]+$ ]] && (( LAST < TARGET_EPOCHS - 1 )); then
        echo "epoch $LAST < $((TARGET_EPOCHS-1)) -- resubmitting to continue"
        cd "$REPO" && SEED=$SEED sbatch scripts/gautschi/submit_o4.sbatch
    else
        echo "seed $SEED reached epoch $LAST -- campaign complete"
    fi
fi
