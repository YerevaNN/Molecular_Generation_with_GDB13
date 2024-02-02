#!/bin/bash

#SBATCH --job-name=MolgenGen
#SBATCH --cpus-per-task=51
#SBATCH --gres=gpu:0
#SBATCH --mem=20gb
#SBATCH --time=24:20:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err
#SBATCH --nice=1


export MODEL="OPT_1.2B_ep_1_half_rand_end_sf_848M_2.00E-04_hf_gradacc_32"
export ITERATION=10000
export PROMPT_FILE="../src/data/data/data_bin_half_rand_sf_848M/valid/00/valid_half_rand_sf_10K.jsonl"


accelerate launch --config_file ../accelerate_cpu_config.yaml \
     ../src/generate_from_valid.py \
    --tokenizer_name ../src/data/tokenizers/tokenizer_sf/tokenizer.json \
    --resume_from_checkpoint ../src/checkpoints/$MODEL/checkpoint-$ITERATION \
    --output_dir ../src/ablations/generations/$MODEL"_"iter_$ITERATION"_"from_valid.csv \
    --batch_size 1 \
    --prompt_file $PROMPT_FILE \
    --num_workers 50 \