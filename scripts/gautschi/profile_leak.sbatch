#!/bin/bash -l
# Bisect the ~170 MB/epoch GPU leak seen on the Gautschi L40S (and NOT on the AF
# A100). Each mode runs as its OWN process so memory state cannot carry across;
# the printed MB/epoch slopes are directly comparable.
#SBATCH --job-name=leakprof
#SBATCH -A cms
#SBATCH -p smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=logs/leakprof_%j.out
#SBATCH --error=logs/leakprof_%j.err

set -uo pipefail

DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
SCRATCH=/scratch/gautschi/das214
REPO=$DEPOT/smart-pixels-ml
VENV=$SCRATCH/envs/smartpix-2bit
OUT=$REPO/runs/leak_profile

mkdir -p logs "$OUT"
module --ignore_cache load modtree/cpu
module --ignore_cache load python/3.11.9
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export SMARTPIX_TFR=$SCRATCH/TFR_files_all101_noise_contained_discovery
export TF_CPP_MIN_LOG_LEVEL=1
export SMARTPIX_GPU_MEM_MB=20000     # cap so a leak shows up fast and bounded

echo "=== $(date) | $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
for mode in sub_lambda sub_fwd sub_fwd_numpy; do
    echo "############ mode=$mode ############"
    python scripts/distillation/profile_leak.py --mode "$mode" --epochs 40 --out "$OUT" || true
done

echo "=== SUMMARY ==="
grep -h "^RESULT" logs/leakprof_${SLURM_JOB_ID}.out 2>/dev/null || true
