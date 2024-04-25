#!/bin/bash

#SBATCH --job-name=CalcStats                     
#SBATCH --cpus-per-task=1        
#SBATCH --mem=20gb                  
#SBATCH --time=01:00:00                          
#SBATCH --output=logging/%x_%j.out  
#SBATCH --error=logging/%x_%j.err

export SUBSET='aspirin'
export SUBSET_LENGTH=8284280
# SUBSET_LENGTH = 8284280 #aspirin
# SUBSET_LENGTH = 6645440 #sas
# SUBSET_LENGTH = 5289763 #druglike
# SUBSET_LENGTH = 5702826 #eqdist

export REPR='selfies'
export TEMPERATURE=1

export PRETRAIN='canon'
export FINETUNE='canon'

export VALID_PROBS_CSV="nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/ablations/perplexities/correct_validations/${REPR}/valid_${PRETRAIN}_${FINETUNE}/${SUBSET}/probs.csv"
export VALID_LENGTH=10000

export GEN_LEN=1000000
export GEN_LEN_STR='1M'
export FROM_GEN='10M'
export GEN_ACTUAL_XLSX="nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/statistics/actual_generation_statistics/${REPR}/${SUBSET}/PRETRAIN_${PRETRAIN}_FINETUNE_${FINETUNE}_FROM_GEN_${FROM_GEN}_GEN_LEN_${GEN_LEN}_TEMP_${TEMPERATURE}.xlsx"

export OUT_PATH="nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/statistics/predicted_generation_statistics/${REPR}/{SUBSET}/PRETRAIN_${PRETRAIN}_FINETUNE_${FINETUNE}_GEN_LEN_${GEN_LEN_STR}_VALID_LEN_${VALID_LENGTH}_TEMP_${TEMPERATURE}.xlsx"
    
python /nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/utils/Metrics_modeling.py \
    --subset $SUBSET \
    --subset_length $SUBSET_LENGTH \
    --valid_probs_csv $VALID_PROBS_CSV \
    --valid_length $VALID_LENGTH \
    --gen_actual_xlsx $GEN_ACTUAL_XLSX \
    --gen_length $GEN_LEN \
    --out_path $OUT_PATH 