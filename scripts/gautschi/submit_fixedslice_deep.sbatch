#!/bin/bash -l
# ---------------------------------------------------------------------------
# O17 DEEP-HEAD ABLATION on frozen slices: is the shallow regression head
# (Flatten(1280) -> Dense(64) -> 14, one hidden layer holding 19.6% of the model
# in a single shared bottleneck) what makes the deep NLL basins rare?
#
# Baseline (same seeds, same frozen [11,26], SHALLOW head): 1 of 5 escaped seeds
# reached NLL -37,255 / cotA -1.878; the rest clustered at -0.18..-0.32. The user
# has seen ~-50K basins on this pair. If the deep head (1280->256->128->64->14,
# +68% params, ViT_MaxDeep_SimpleRouter) lifts the hit rate, head capacity was
# limiting RELIABILITY; if the scatter stays 1-in-5, the head is exonerated and
# the landscape/objective story stands alone. Uncapped constraints ON PURPOSE --
# identical to the baseline arm; do not add LAMBDA/INF caps here.
#
# Every run so far conflates those two questions. Only 1 of 8 O11 seeds picked an
# early pair ([19,20], seed 2042) and it beat every late-slice seed on cot alpha
# by 0.36 nats -- suggestive, but n=1. Here the router is PINNED (theta frozen,
# verified: mu = 1.0 on each chosen slice, training sampler visits only those
# two, theta absent from trainable_variables), so the only thing varying is which
# slices the network is given.
#
# Same O13 two-lambda angle pressure on every arm, so each pair is pushed on the
# angles equally and the comparison is "best cot alpha this pair can support".
#
# Same SEEDS across all three pairs => paired comparison, init held constant.
#
#   PAIR=19,20 SEEDS=41042,41142,41242 sbatch scripts/gautschi/submit_fixedslice_deep.sbatch
#
# ONE JOB PER PAIR, all three seeds sequentially. One-job-per-seed thrashed:
# smallgpu nodes carry 2 GPUs, sacct showed THREE of my jobs on g005 at once, each
# claiming a 20 GB TF pool on a 46 GB card, so they OOM-evicted each other every
# ~2 epochs (7-minute jobs advancing 2 epochs, then resubmitting). 2000 epochs at
# ~6 s/epoch is ~3.3 h per seed, so 3 seeds fit inside the 12 h cap -- and 3 jobs
# instead of 9 stops the QOS submit limit mattering. Memory cap also dropped to
# 14 GB so a co-tenant cannot starve us.
#
# 2000 epochs, not 5000: O12 reached its angle plateau by ~epoch 800, and this is
# a comparison between arms rather than a production run. All arms get the same
# budget, so it stays internally consistent.
# ---------------------------------------------------------------------------
# job name is set at submit time (-J fx_<pair>_<seed>) so squeue shows the arm
#SBATCH -A cms
#SBATCH -p smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/fix_%x_%j.out
#SBATCH --error=logs/fix_%x_%j.err
set -euo pipefail
DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
SCRATCH=/scratch/gautschi/das214
REPO=$DEPOT/smart-pixels-ml
VENV=$SCRATCH/envs/smartpix-2bit
SEEDS=${SEEDS:?set SEEDS=a,b,c}
PAIR=${PAIR:?set PAIR=a,b}
TARGET_EPOCHS=2000
TAG=$(echo "$PAIR" | tr ',' '_')
OUT=$REPO/runs/fixedslice_deep_p$TAG
mkdir -p logs "$OUT"
module --ignore_cache load modtree/cpu
module --ignore_cache load python/3.11.9
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export SMARTPIX_TFR=$SCRATCH/TFR_files_all101_noise_contained_discovery
export TF_CPP_MIN_LOG_LEVEL=1
export SMARTPIX_GPU_MEM_MB=14000
export SMARTPIX_FIX_SLICES=$PAIR
export SMARTPIX_CONSTRAINT=anglenll
export SMARTPIX_NLL_BLOCK='cotA,cotB'
export SMARTPIX_ANGLE_NLL_TARGET='-0.15,-2.20'
export SMARTPIX_LOSS_CLIP=1
export SMARTPIX_MODEL_NAME=ViT_MaxDeep_SimpleRouter
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-none} | pair [$PAIR] seeds $SEEDS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$REPO"
# `|| true`: this GPU leaks ~170 MB/epoch so a chunk is EXPECTED to OOM partway;
# resume is exact, so the chain grinds forward.
NSEEDS=$(echo "$SEEDS" | tr ',' ' ' | wc -w)
python scripts/distillation/run_simplerouter_mdmm_discovery.py \
    --epochs "$TARGET_EPOCHS" --seeds "$SEEDS" --extend --target "$NSEEDS" --out "$OUT" || true
# done only when EVERY seed of this pair has reached the target
DONE=1
for S in $(echo "$SEEDS" | tr ',' ' '); do
    H=$OUT/seed_$S/history.csv
    L=0; [[ -f $H ]] && L=$(tail -1 "$H" | cut -d, -f1 | tr -d '\r')
    [[ "$L" =~ ^[0-9]+$ ]] || L=0
    echo "  seed $S at epoch $L / $TARGET_EPOCHS"
    (( L < TARGET_EPOCHS - 1 )) && DONE=0
done
if (( ! DONE )); then
    echo "pair [$PAIR] incomplete -- resubmitting"
    cd "$REPO" && SEEDS=$SEEDS PAIR=$PAIR sbatch -J "fxD_${TAG}" scripts/gautschi/submit_fixedslice_deep.sbatch
else
    echo "pair [$PAIR] complete"
fi
# ---- drain the pending queue -----------------------------------------------
# Gautschi caps submitted jobs per user (QOSMaxSubmitJobPerUserLimit, 8 here), so
# the full grid cannot be queued up front. Each ARM THAT FINISHES pops the next
# combo, under flock so two finishing jobs cannot take the same line.
PENDING=$REPO/runs/fixedslice_deep_pending.txt
if (( DONE )) && [[ -f $PENDING ]]; then
    exec 9>"$PENDING.lock"
    if flock -w 60 9; then
        NEXT=$(head -1 "$PENDING" || true)
        if [[ -n "$NEXT" ]]; then
            tail -n +2 "$PENDING" > "$PENDING.tmp" && mv "$PENDING.tmp" "$PENDING"
            NP=${NEXT%% *}; NS=${NEXT##* }
            echo "draining queue -> pair [$NP] seeds $NS"
            cd "$REPO" && PAIR=$NP SEEDS=$NS sbatch -J "fxD_$(echo "$NP" | tr "," "_")" \
                scripts/gautschi/submit_fixedslice_deep.sbatch || true
        fi
        flock -u 9
    fi
fi
