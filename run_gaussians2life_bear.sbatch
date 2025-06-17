#!/bin/bash
#SBATCH --job-name="gaussiuanss2life_bear_run1"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=20G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=neil.de@tum.de
#SBATCH --output=/storage/user/deln/projects/threestudio-1/slurm/logs/bear/slurm-%j.out
#SBATCH --error=/storage/user/deln/projects/threestudio-1/slurm/logs/bear/slurm-%j.err
#SBATCH --constraint="GPU_CC:8.9"

# ---------- Conda environment ----------
eval "$(/storage/user/deln/miniconda3/bin/conda shell.bash hook)"
conda activate gaussians2life

echo "Host: $(hostname)  |  Dir: $(pwd)"
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Default GPU index if $GPU isn’t set
GPU=${GPU:-0}

PROMPT="bear statue turns its head"
# Slugify: spaces → _, drop non-alphanum/_ 
TAG=$(echo "$PROMPT" \
      | tr '[:upper:]' '[:lower:]' \
      | tr ' ' '_' \
      | tr -cd '[:alnum:]_')

echo "Prompt: $PROMPT"
echo "Tag:    $TAG"


pwd; hostname; date

# Run
srun python launch.py \
     --config custom/gaussians2life/configs/bear.yaml \
     --train \
     --gpu "${GPU:-0}" \
     system.prompt_processor.prompt="$PROMPT" \
     tag="$TAG" 