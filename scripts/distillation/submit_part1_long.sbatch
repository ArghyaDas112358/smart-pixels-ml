#!/bin/bash
#SBATCH --job-name=p1long
#SBATCH --partition=hammer-gpu
#SBATCH --account=cms-express
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/work/users/das214/SmartPixels/smart-pixels-ml/runs/part1_long/slurm/%x_%A_%a.out
#SBATCH --error=/work/users/das214/SmartPixels/smart-pixels-ml/runs/part1_long/slurm/%x_%A_%a.err

PY=/work/users/das214/envs/smartpix-2bit/bin/python
SEED=$((42 + SLURM_ARRAY_TASK_ID * 1000))
cd /work/users/das214/SmartPixels/smart-pixels-ml
echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES seed=$SEED task=$SLURM_ARRAY_TASK_ID"
$PY -u scripts/distillation/run_part1_long.py --seed "$SEED" --epochs 1000 --out runs/part1_long
