#!/bin/bash

#SBATCH --job-name=ACE_dataset_generation
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=10gb
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/vampire/%x_%j.out
#SBATCH --error=logging/vampire/%x_%j.err

#SBATCH --partition=all

# Get directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add project root to Python path
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH}"


python "${SCRIPT_DIR}/../src/data/ACE_data/generate.py" \
    --output_dir /nfs/h100/raid/chem \
    --depth 18