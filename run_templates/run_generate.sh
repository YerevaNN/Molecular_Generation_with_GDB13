#!/bin/bash

#SBATCH --job-name=GenAsp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err

export SUBSET='sas'
export SUBSET_EXT='sas_3'
export STR_TYPE='selfies'
export SHORT_STR_TYPE='sf'

export GEN_LEN=1000000
export GEN_LEN_STR='1M'
export PROMPT="_" 

export TOP_K=192
export TOP_P=1.0
export TEMPERATURE=1.0

export ITERATION=3900
export LR='16.00E-05'
export PRETRAIN='canon'
export FINETUNE='rand'

export TOKENIZER="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/data/tokenizers/tokenizer_${SHORT_STR_TYPE}/tokenizer.json"
export MODEL="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/checkpoints/main_checkpoints/OPT_1.2B_ep_1_all_${PRETRAIN}_finetune_all_${FINETUNE}_${SUBSET_EXT}_sf_1000K_${LR}_hf/checkpoint-${ITERATION}"
export OUT_PATH="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/generations/${STR_TYPE}/${SUBSET}/gen_len_${GEN_LEN_STR}_p_${TOP_P}_k_${TOP_K}_temp_${TEMPERATURE}_${PRETRAIN}_${FINETUNE}.csv"

accelerate launch --config_file /nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/accelerate_gen_config.yaml \
     /nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/generate.py \
     --tokenizer_name $TOKENIZER\
     --resume_from_checkpoint $MODEL \
     --output_dir  $OUT_PATH\
     --top_k $TOP_K \
     --top_p $TOP_P \
     --temperature $TEMPERATURE \
     --prompt_token $PROMPT \
     --gen_len $GEN_LEN \
     --batch_size 4096
