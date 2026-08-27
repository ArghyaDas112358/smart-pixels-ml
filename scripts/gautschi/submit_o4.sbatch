#!/bin/bash -l
# NOTE the `-l`: Slurm runs batch scripts in a NON-login shell, where Lmod is not
# initialised and `module` is simply not a command (job 14497092 died in 1s on
# exactly this). A login shell sources /etc/profile.d and makes module work.
# ---------------------------------------------------------------------------
# O4 on Gautschi: MDMM x SimpleRouter with an ANNEALED pair-distribution
# temperature (beta 1 -> 30). This is the fix for the router never committing:
# on Purdue AF, mu stayed spread over ~60 effective slices even after 2500
# epochs, so one weight set and ONE SoftQuantize threshold triple had to serve
# amplitudes from 74 to 335 mV -- only the cluster centroid survives that, which
# is why positions sit at 0.99 while angles mean-revert.
#
# PARTITION CHOICE: smallgpu (L40 48GB, 12h cap), NOT ai (H100). The cms and
# physics accounts both show "GPU Hours Balance 0.0" on the AI partition, and a
# test submit there fails with AssocGrpBillingMinutes under every QOS. smallgpu
# requires 64 CPUs per GPU (the scheduler rejects other ratios).
#
# DATA lives on Lustre scratch, not Depot: this job reads ~16 GB/epoch and the
# Depot NFS mount cannot sustain that (a conda build alone hung every /depot stat
# on the login node for minutes).
#
# Submit a 3-epoch smoke test:   SANITY=1 sbatch scripts/gautschi/submit_o4.sbatch
# Submit the real run:                  sbatch scripts/gautschi/submit_o4.sbatch
# Status:                               squeue --me
#
# SELF-CHAINING: the 12h cap is far shorter than a 5000-epoch run, so the job
# resubmits itself until the seed reaches the target epoch. Safe because the
# driver resumes exactly: last.weights.hdf5 every epoch, Lagrange multipliers
# replayed from mdmm_epochs.csv, and a FIXED 1000-epoch anneal horizon so a
# resumed run never re-softens an already-annealed quantizer.
# KNOWN GAP: resume does NOT restore the Nadam optimizer moments, so a 5000-epoch
# run split into 12h chunks eats ~8 optimizer resets.
# ---------------------------------------------------------------------------
#SBATCH --job-name=o4router
#SBATCH -A cms
#SBATCH -p smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/o4_%x_%j.out
#SBATCH --error=logs/o4_%x_%j.err

set -euo pipefail

DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
SCRATCH=/scratch/gautschi/das214
REPO=$DEPOT/smart-pixels-ml
VENV=$SCRATCH/envs/smartpix-2bit

SEED=${SEED:-8042}                  # a seed NOT used by the AF O11 campaign
SANITY=${SANITY:-0}
TARGET_EPOCHS=5000
BETA_FINAL=30                       # top-2 mu mass ~1.66 at beta=20, ~1.97 at 60
BETA_START=1000                     # hold beta=1 until the quantizer anneal ends
BETA_EPOCHS=2000                    # then ramp over 2000 epochs, then hold
OUT=$REPO/runs/simplerouter_mdmm_o4beta   # SEPARATE dir: beta checkpoints are
                                          # incompatible with plain-router ones

mkdir -p logs "$OUT"

module --ignore_cache load modtree/cpu
module --ignore_cache load python/3.11.9
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# read the dataset from Lustre, not the AF /work paths baked into the driver
export SMARTPIX_TFR=$SCRATCH/TFR_files_all101_noise_contained_discovery
export TF_CPP_MIN_LOG_LEVEL=1
# Job 14497729 OOM'd at epoch 311 on a 46 GB L40S despite a ~5 GB working set --
# i.e. heap FRAGMENTATION in TF's default BFC allocator, not a real shortage
# (the same code runs 4-up on a 40 GB A100 for days). The async CUDA allocator
# returns freed blocks to the driver and does not fragment the same way.
# TF_GPU_ALLOCATOR=cuda_malloc_async REMOVED: the async allocator IGNORES the
# logical-device memory_limit below, so the two settings fought and the pool ate
# the whole card (job 14609486 died with CUDA reporting 99 MB free of 47.7 GB,
# allocator name "gpu_async_0"). The default BFC allocator DOES respect the cap.
export SMARTPIX_GPU_MEM_MB=20000   # hard 20 GB pool: growth alone OOMd at ep 311 and 563

echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-none} | seed $SEED | sanity=$SANITY ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import tensorflow as tf; print('GPUs visible to TF:', tf.config.list_physical_devices('GPU'))"

cd "$REPO"
if [[ "$SANITY" == "1" ]]; then
    python scripts/distillation/run_simplerouter_mdmm_discovery.py \
        --sanity --seeds "$SEED" --extend --target 1 \
        --beta-final "$BETA_FINAL" --out "$OUT"
    echo "SANITY_DONE"
    exit 0
fi

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
