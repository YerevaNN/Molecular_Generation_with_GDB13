#!/bin/bash

#SBATCH --job-name=Evaluate
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err

export SAVE_TYPE='CANON'
export STR_TYPE='selfies'
export SHORT_STR_TYPE='sf'

export SUBSET='aspirin'
export SUBSET_EXT='aspirin_0.4'

export PRETRAIN='canon'
export FINETUNE='canon'
export LR='8.00E-05'
export ITERATION=3900

export TOP_K=192
export TOP_P=1.0
export TEMPERATURE=1.0

export DATA="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13_my/src/data/data/randomized_valids/${STR_TYPE}/${SUBSET}/rand_valid_10K_sf.jsonl"
export TOKENIZER="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/data/tokenizers/tokenizer_${SHORT_STR_TYPE}/tokenizer.json"
export CHECKPOINT="OPT_1.2B_ep_1_all_${PRETRAIN}_finetune_all_${FINETUNE}_${SUBSET_EXT}_${SHORT_STR_TYPE}_1000K_${LR}"
export MODEL="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/checkpoints/fine_tuned/${CHECKPOINT}/checkpoint-${ITERATION}"
export OUT_PATH="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/ablations/perplexities/correct_validations/${STR_TYPE}/valid_${PRETRAIN}_${FINETUNE}/${SUBSET}/top_k_${TOP_K}_top_p_${TOP_P}_temperature_${TEMPERATURE}.csv"

accelerate launch --config_file /nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/accelerate_eval_config.yaml \
     /nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/validate.py \
     --data_path $DATA \
     --tokenizer_path $TOKENIZER \
     --resume_from_checkpoint $MODEL \
     --top_k $TOP_K \
     --top_p $TOP_P \
     --temperature $TEMPERATURE \
     --save_type $SAVE_TYPE \
     --str_type $STR_TYPE \
     --output_path $OUT_PATH \
     --batch_size 128