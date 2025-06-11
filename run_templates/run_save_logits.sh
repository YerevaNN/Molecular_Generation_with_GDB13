#!/bin/bash

#SBATCH --job-name=SaveLogits
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err

export CUDA_VISIBLE_DEVICES=1

export STR_TYPE='sf'

export SUBSET='aspirin'
export SUBSET_EXT='aspirin_0.4'

export DATA="../src/data/data/randomized_valids/${STR_TYPE}/${SUBSET}/rand_valid_10K_sf.jsonl"
export TOKENIZER="../src/data/tokenizers/tokenizer_${STR_TYPE}/tokenizer.json"
export MODEL="../src/checkpoints/fine_tuned/sf/OPT_1.2B_ep_1_all_canon_finetune_all_canon_${SUBSET_EXT}_${STR_TYPE}_1000K_8.00E-05/checkpoint-3900"
export OUT_PATH="../src/data/data/randomized_valids/${STR_TYPE}/${SUBSET}/logits/valid_OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_test.npy"

accelerate launch --config_file ../accelerate_fsdp_config_1.yaml \
     ../src/save_logits.py \
     --data_path $DATA \
     --tokenizer_path $TOKENIZER \
     --resume_from_checkpoint $MODEL \
     --output_path $OUT_PATH \
     --batch_size 2
     
     