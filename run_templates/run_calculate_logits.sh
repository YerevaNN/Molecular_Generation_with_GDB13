#!/bin/bash

#SBATCH --job-name=CalcLogits
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err


# export CUDA_VISIBLE_DEVICES=1

export SAVE_TYPE='CANON'
export STR_TYPE='sf'

export SUBSET='aspirin'
export SUBSET_EXT='aspirin_0.4'

export TOP_K=192
export TOP_P=1.0
export TEMPERATURE=0.9

for TEMPERATURE in 1.4
do
     export DATA="../src/data/data/randomized_valids/${STR_TYPE}/${SUBSET}/rand_valid_10K_sf.jsonl"
     export TOKENIZER="../src/data/tokenizers/tokenizer_${STR_TYPE}/tokenizer.json"
     export IN_PATH="../src/data/data/randomized_valids/${STR_TYPE}/${SUBSET}/logits/valid_OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_original.npy"
     export OUT_PATH="../src/ablations/perplexities/${STR_TYPE}/temps/valid_OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_givenTemp_${TEMPERATURE}.csv"

     accelerate launch --config_file ../accelerate_fsdp_config_1.yaml \
          ../src/calc_logits.py \
          --data_path $DATA \
          --tokenizer_path $TOKENIZER \
          --top_k $TOP_K \
          --top_p $TOP_P \
          --temperature $TEMPERATURE \
          --save_type $SAVE_TYPE \
          --str_type $STR_TYPE \
          --in_path $IN_PATH \
          --output_path $OUT_PATH \
          --batch_size 8192
done     