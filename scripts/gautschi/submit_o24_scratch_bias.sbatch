#!/bin/bash -l
# NOTE the `-l`: Slurm batch scripts run in a NON-login shell where Lmod is not
# initialised and `module` does not exist.
# ---------------------------------------------------------------------------
# O24 -- the O22 loss, run as a FULL COLD CAMPAIGN.
#
# O22 was a patch: warm-start the converged (and already biased) parent, freeze
# its slice pair and thresholds, and correct the bias for 100 epochs. This asks
# the harder question -- what does the model choose when the bias constraint is
# part of the objective from epoch zero?
#
# Everything matches the parent campaign (submit_o21v2_cold.sbatch) EXCEPT the
# loss: free pair-lattice router, trainable thresholds, no warm start, 10,000
# epochs. Nothing is frozen. The router may well settle on a different pair
# than (10,21) under a bias-aware objective -- that is the point.
#
#   SEED=40342 sbatch scripts/gautschi/submit_o24_scratch_bias.sbatch
#
# MEMORY: SMARTPIX_GPU_MEM_MB caps TF to 20 GB. Do NOT also set
# TF_GPU_ALLOCATOR=cuda_malloc_async -- it ignores the cap.
# ---------------------------------------------------------------------------
#SBATCH --job-name=o24bias
#SBATCH -A cms
#SBATCH -p smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/o24_%x_%j.out
#SBATCH --error=logs/o24_%x_%j.err
set -euo pipefail
DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
SCRATCH=/scratch/gautschi/das214
REPO=$DEPOT/smart-pixels-ml
VENV=$SCRATCH/envs/smartpix-2bit
SEED=${SEED:?set SEED=<n> when submitting}
TARGET_EPOCHS=10000
MODEL=ViT_MaxDeep_PairLattice
# ONE OUTPUT DIR PER SEED. --target counts already-converged seeds in the SAME
# dir, so a shared campaign dir lets the first seed to converge pre-satisfy
# every other chain, which then exits without training and without complaining.
OUT=$REPO/runs/o24_scratch_bias/s$SEED
mkdir -p logs "$OUT"
module --ignore_cache load modtree/cpu
module --ignore_cache load python/3.11.9
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export SMARTPIX_TFR=$SCRATCH/TFR_files_all101_noise_contained_discovery
export TF_CPP_MIN_LOG_LEVEL=1
# 42 GB, not the historical 20 GB: smallgpu nodes are L40S (46 GB physical),
# and the zero-bias constraint roughly TRIPLED the per-epoch graph-memory
# growth -- 20 GB chunks died at ~58 epochs (vs ~155 on the parent campaign),
# i.e. ~172 queue cycles per seed for 10K epochs. The cap exists to keep the
# OOM inside TF (clean resume) rather than the CUDA context; 42 GB preserves
# that with ~4 GB headroom while doubling the epochs each chunk survives.
export SMARTPIX_GPU_MEM_MB=42000
export SMARTPIX_MODEL_NAME=$MODEL
export SMARTPIX_LOSS_V2=1
export SMARTPIX_ABORT_THR=1e12
export SMARTPIX_SMOOTH_SIGMA0=4
export SMARTPIX_SMOOTH_EPOCHS=3000
# ---- the ONLY difference from the parent campaign: the O22 loss -------------
export SMARTPIX_ZEROBIAS=1
export SMARTPIX_BINBAL=byterm
export SMARTPIX_NBINS=15
export SMARTPIX_BIAS_TOL=0.17
export SMARTPIX_BIAS_MAXLAM=5.0
export SMARTPIX_BIAS_INFCAP=2.0
# DELIBERATELY NOT SET: SMARTPIX_WARM_START, SMARTPIX_FREEZE_THR,
# SMARTPIX_PIN_MODE. Cold, free router, trainable thresholds -- as the parent.
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-none} | O24 COLD+BIAS seed $SEED ==="
echo "    warm start: NONE | loss: v2 + byterm binbal + zerobias(tol 0.17) | epochs: $TARGET_EPOCHS"
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
        # RESUBMIT GUARD: sbatch can be REFUSED (QOS job cap). Ignoring its exit
        # status once left two seeds dead for hours. Retry, then shout.
        for _try in 1 2 3 4 5; do
            if cd "$REPO" && SEED=$SEED sbatch scripts/gautschi/submit_o24_scratch_bias.sbatch; then
                echo "resubmit ok (attempt $_try)"; break
            fi
            echo "RESUBMIT FAILED (attempt $_try) -- retrying in 120s"
            if [ "$_try" = 5 ]; then echo "RESUBMIT GUARD: GIVING UP -- CHAIN DEAD"; else sleep 120; fi
        done
    else
        echo "seed $SEED reached epoch $LAST -- done"
    fi
fi
