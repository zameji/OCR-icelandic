#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=smolVLM_lora_icelandic_v1
#SBATCH --output=logs/smolVLM_lora_icelandic_v1-%j.log
#SBATCH --error=logs/smolVLM_lora_icelandic_v1-%j.err
#SBATCH --gpus-per-node=1
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
EXPERIMENT_NAME="smolVLM-lora-icelandic-v1"
MODEL_VERSION="base"
DATASET_VERSION="igc2024-wiki-1pct"
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

# 1648 training steps
python3 scripts/train_llm.py \
    push_to_hub=True \
    logging_steps=100 \
    save_steps=200 \
    eval_steps=200 \
    run_name="smolVLM_icelandic_igc2024-wiki"


# Log completion
echo "=== Job Completion ==="
echo "End Time: $(date)"
echo "======================="