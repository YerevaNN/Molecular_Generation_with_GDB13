#!/bin/bash

# Define the start and increment values
# all lines = 423885787
start_line=0
increment=10000000
input_file="./data_bin_all_sf_848M/train/00/train_all_sf_848M.jsonl"
output_file="./rand_all_"
prefix=""
suffix=""

for i in {1..10}; do
  # Run the Python script with the current start line and num_lines in the background
  python prepare_pretrain_data.py --start $start_line --increment $increment --input_file "$input_file" --output_file "$output_file" --prefix "$prefix" --suffix "$suffix" --rand &
  
  # Increment the start_line for the next iteration
  start_line=$((start_line + increment))
done

# Wait for all background jobs to finish
wait
echo "All processing jobs are completed."