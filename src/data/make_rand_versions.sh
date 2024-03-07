#!/bin/bash

export rand_number=8
export train_set="equal_dist"

for train_set in "equal_dist"
do
input_file="./data/data_bin_all_rand_$train_set"_"sf_1000K/train/00/train_all_rand_$train_set"_"sf_1000K.jsonl"
output_file="./data/data_bin_all_rand_$train_set"_"sf_1000K_rand_$rand_number"_"versions/train/00/train_all_rand_$train_set"_"sf_1000K_rand_$rand_number"_"versions.jsonl"

python make_rand_versions.py --rand_number $rand_number --input_file "$input_file" --output_file "$output_file"
done  

# Wait for all background jobs to finish
wait
echo "All processing is completed."