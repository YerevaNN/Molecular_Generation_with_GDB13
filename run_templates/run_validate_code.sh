#!/bin/bash

#SBATCH --job-name=ValidateCodeToken_new_16
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=70gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err
#SBATCH --partition=all


export CUDA_VISIBLE_DEVICES=1


export TOP_K=128256
export TOP_P=1.0
export TEMPERATURE=1

export TOKENIZER="meta-llama/Llama-3.2-1B"
export MODEL="/auto/home/knarik/Molecular_Generation_with_GDB13/src/checkpoints/checkpoints_code/Llama-3-1B_tit_hf_4_epochs/step-3126"
export DATA="../src/data/data/data_bin_python/valid_random/random_val.jsonl"
export OUT_PATH="../src/ablations/perplexities/code/Llama3.2_1B_ep_4.csv"
 
accelerate launch --config_file ../accelerate_local_config.yaml \
     ../src/validate_code.py \
     --data_path $DATA \
     --tokenizer_path $TOKENIZER \
     --resume_from_checkpoint $MODEL \
     --top_k $TOP_K \
     --top_p $TOP_P \
     --temperature $TEMPERATURE \
     --output_path $OUT_PATH \
     --batch_size 512