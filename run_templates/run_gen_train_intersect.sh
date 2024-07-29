#!/bin/bash

#SBATCH --job-name=AspirinIntersect
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=20gb
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err

export repr="sf"

for data_set in "aspirin_0.4"
do
    python ../src/utils/gen_train_intersection.py \
    --gen_file "../src/ablations/generations/generations/$repr/OPT_1.2B_ep_1_all_canon_finetune_all_canon_$data_set"_"$repr"_"1000K_8.00E-05_hf_iter_3900_gen_1000000__.csv" \
    --train_file "../src/data/data/data_bin_all_canon_$data_set"_"$repr"_"1000K/train/00/train_all_canon_$data_set"_"$repr"_"1000K.jsonl" \
    --repr $repr
done