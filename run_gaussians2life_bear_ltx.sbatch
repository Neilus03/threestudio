#!/bin/bash
#SBATCH --job-name="g2l_bear_ltx_run1"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=160GB
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com
#SBATCH --output=/home/stud/deln/storage/user/projects/threestudio-1/slurm/logs/bear_ltx/slurm-%j.out
#SBATCH --error=/home/stud/deln/storage/user/projects/threestudio-1/slurm/logs/bear_ltx/slurm-%j.err
#SBATCH --constraint="GPU_CC:8.9"

# -------------------------------------------------------------------
#  Conda environment
# -------------------------------------------------------------------
eval "$(/storage/user/deln/miniconda3/bin/conda shell.bash hook)"   # <- adjust if Miniconda lives elsewhere
conda activate gaussians2life                                      # env must include LTX-Video deps

# -------------------------------------------------------------------
echo "Host: $(hostname)  |  Dir: $(pwd)"
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# -------------------------------------------------------------------

GPU=${GPU:-0}

PROMPT="The statue bear lifts both frontpaws and falls down."
TAG_BASE=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd '[:alnum:]_')
TAG="ltx_${TAG_BASE}"

echo "Prompt: $PROMPT"
echo "Tag:    $TAG"

# create log directory if it isn’t there
LOGDIR=/home/stud/deln/storage/user/projects/threestudio-1/slurm/logs/bear_ltx
mkdir -p "$LOGDIR"

# reduce CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/stud/deln/storage/user/projects/threestudio-1   # threestudio root

# -------------------------------------------------------------------
#  Launch Gaussians-to-Life with LTX-Video guidance
# -------------------------------------------------------------------
srun python launch.py \
    --config custom/gaussians2life/configs/bear_ltx.yaml \
    --train \
    --gpu "$GPU" \
    system.guidance.prompt="$PROMPT" \
    tag="$TAG"
