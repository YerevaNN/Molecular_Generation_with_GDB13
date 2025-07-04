#!/bin/bash

#SBATCH --job-name=Validate
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err


export CUDA_VISIBLE_DEVICES=1

export SAVE_TYPE='CANON'
export STR_TYPE='sf'

export SUBSET='aspirin'
export SUBSET_EXT='aspirin_0.4'

export TOP_K=192
export TOP_P=1.0
export TEMPERATURE=0.8

for TEMPERATURE in 1
do
     export DATA="../src/data/data/randomized_valids/${STR_TYPE}/${SUBSET}/rand_valid_10K_sf.jsonl"
     # export DATA="../src/ablations/generations/generations/sf/temperatures/OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_gen_1M_temp_${TEMPERATURE}.csv"
     export TOKENIZER="../src/data/tokenizers/tokenizer_${STR_TYPE}/tokenizer.json"
     export MODEL="../src/checkpoints/fine_tuned/sf/OPT_1.2B_ep_1_all_canon_finetune_all_canon_${SUBSET_EXT}_${STR_TYPE}_1000K_8.00E-05/checkpoint-3900"
     export OUT_PATH="../src/ablations/perplexities/${STR_TYPE}/temps/valid_OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_givenTemp_${TEMPERATURE}_for_compare.csv"

     accelerate launch --config_file ../accelerate_fsdp_config_1.yaml \
          ../src/validate.py \
          --data_path $DATA \
          --tokenizer_path $TOKENIZER \
          --resume_from_checkpoint $MODEL \
          --top_k $TOP_K \
          --top_p $TOP_P \
          --temperature $TEMPERATURE \
          --save_type $SAVE_TYPE \
          --str_type $STR_TYPE \
          --output_path $OUT_PATH \
          --batch_size 8192
done     