#!/bin/bash

#SBATCH --job-name=Equiv3.2_depth_8_var_3
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:0
#SBATCH --mem=80gb
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/python/%x_%j.out
#SBATCH --error=logging/python/%x_%j.err


python ../src/data/Python_data/equivalence_check_parallel.py \
    --path "/nfs/h100/raid/chem/exhaustive_generations/data_level_3.2_depth_8_3_vars.txt"