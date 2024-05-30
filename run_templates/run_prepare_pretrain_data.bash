#!/bin/bash

#SBATCH --job-name=RandDatasetForSmiles
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:0
#SBATCH --mem=100gb
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err

# Define the start and increment values
# all lines = 423885787
start_line=0
increment=10000
repr="sm"
input_file="../src/data/data/data_bin_all_canon_sm_848M/valid/00/valid_all_canon_sm_10K.jsonl"
output_file="../src/data/data/rand_all_$repr"_
prefix=""
suffix=""


for i in {0..2}; do
  # Run the Python script with the current start line and num_lines in the background
  python "../src/data/prepare_pretrain_data.py" --start $start_line --increment $increment --repr $repr --input_file "$input_file" --output_file "$output_file" --prefix "$prefix" --suffix "$suffix" --rand &
  
  # Increment the start_line for the next iteration
  start_line=$((start_line + increment))
done

# # Wait for all background jobs to finish
# wait

# python  ../src/data/prepare_pretrain_data.py --start $start_line --increment $increment --input_file "$input_file" --output_file "$output_file" --prefix "$prefix" --suffix "$suffix" --rand
  
echo "All processing jobs are completed."