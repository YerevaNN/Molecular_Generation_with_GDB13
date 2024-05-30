#!/bin/bash

#SBATCH --job-name=RandToCanon
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:0
#SBATCH --mem=100gb
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err

# Define the start and increment values
start_line=800000000
increment=1000000
input_file="../src/data/data/train_rand_all_sm_848M_no_empty.jsonl"
output_file="../src/data/data/canon_all_"

for i in {0..99}; do
    python "../src/utils/rand_to_canon_parallel.py" --start $start_line --increment $increment --input_file "$input_file" --output_file "$output_file" &
    
    start_line=$((start_line + increment))
done