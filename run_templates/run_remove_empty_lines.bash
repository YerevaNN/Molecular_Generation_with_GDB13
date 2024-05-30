#!/bin/bash

#SBATCH --job-name=RemoveEmptyLines
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:0
#SBATCH --mem=100gb
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err


python "../src/data/remove_empty_lines.py"