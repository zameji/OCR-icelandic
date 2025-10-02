#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=generate_synthetic_data
#SBATCH --output=logs/generate_synthetic_data_v1-%j.log
#SBATCH --error=logs/generate_synthetic_data_v1-%j.err
#SBATCH --cpus-per-task=1

set -e  # Exit on any error

SCRATCH=/scratch/s5982960/OCR-icelandic

module purge
module load Python/3.13.1-GCCcore-14.2.0

mkdir -p $SCRATCH

RUN_NAME="Generate Synthetic Data"

# Log system info
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Run Name: $RUN_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "========================"

# Copy data to scratch if needed

# Copy data from scratch

# run our dataset script
cd $SCRATCH
source $SCRATCH/.venv/bin/activate
uv sync


python3 prepare_data.py max_entries=0 dataset_path="arnastofnun/IGC-2024" text_column="document" data_directory="wiki" push_to_hub=True

# Log completion
echo "=== Job Completion ==="
echo "End Time: $(date)"
echo "======================="

