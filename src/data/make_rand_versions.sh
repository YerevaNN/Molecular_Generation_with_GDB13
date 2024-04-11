#!/bin/bash

export rand_number=8
export split="valid"
export count="10K"

for data_set in "aspirin_0.4" "druglike_0.4" "sas_3" "equal_dist"
do
    input_file="./data/data_bin_all_rand_$data_set"_"sf_1000K_rand_8_versions_valid_80K/$split/00/$split"_"$data_set"_"sf_$count.jsonl"
    output_file="./data/data_bin_all_rand_$data_set"_"sf_1000K_rand_$rand_number"_"versions_valid_80K/$split/00/$split"_"all_rand_$data_set"_"sf_$count"_"rand_$rand_number"_"versions.jsonl"

    python make_rand_versions.py --rand_number $rand_number --input_file "$input_file" --output_file "$output_file"
done  

echo "All processing is completed."