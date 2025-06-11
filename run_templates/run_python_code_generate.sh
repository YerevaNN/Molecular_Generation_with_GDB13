#!/bin/bash

#SBATCH --job-name=Code3.2_depth_8_var_3_after_fix
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=40gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/python/%x_%j.out
#SBATCH --error=logging/python/%x_%j.err

#SBATCH --partition=all


python ../src/data/Python_data/code_generator.py