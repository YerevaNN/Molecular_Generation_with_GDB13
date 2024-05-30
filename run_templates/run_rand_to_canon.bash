#!/bin/bash

#SBATCH --job-name=RandToCanon
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:0
#SBATCH --mem=100gb
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err

input_file="../src/data/data/data_bin_all_rand_sf_848M/valid/00/valid_all_rand_sf_10K.jsonl"
output_file="../src/data/data/valid_all_canon_sm_10K.jsonl"

python "../src/utils/rand_to_canon.py" --input_file "$input_file" --output_file "$output_file"