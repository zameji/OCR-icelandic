#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --job-name=smolVLM_lora_icelandic_v3
#SBATCH --output=logs/smolVLM_lora_icelandic_v3-%j.log
#SBATCH --error=logs/smolVLM_lora_icelandic_v3-%j.err
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e  # Exit on any error

SCRATCH=/scratch/s5982960/OCR-icelandic
# export HF_HOME=/tmp

module purge
module load CUDA/12.6.0
module load Python/3.13.1-GCCcore-14.2.0
module load Boost/1.79.0-GCC-11.3.0

mkdir -p $SCRATCH

# Experiment configuration
EXPERIMENT_NAME="smolVLM-lora-icelandic-v4"
MODEL_VERSION="base"
DATASET_VERSION="Sigurdur/isl_synthetic_ocr"
LORA_CONFIG="r16-alpha32"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${EXPERIMENT_NAME}-${MODEL_VERSION}-${DATASET_VERSION}-${LORA_CONFIG}-${TIMESTAMP}"

# Log system info
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Run Name: $RUN_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "========================"

# Copy data to scratch if needed
# currently handled by the training script

# Copy data from scratch
# currently handled by the training script

# run our fine-tuning script
cd $SCRATCH
source $SCRATCH/.venv/bin/activate
uv sync

python3 smol_vlm_ft.py


# Log completion
echo "=== Job Completion ==="
echo "End Time: $(date)"
echo "======================="