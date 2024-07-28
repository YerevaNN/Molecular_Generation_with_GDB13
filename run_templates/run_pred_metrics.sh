#!/bin/bash

#SBATCH --job-name=PredMetrics                     
#SBATCH --cpus-per-task=1        
#SBATCH --mem=20gb                  
#SBATCH --time=02:00:00                          
#SBATCH --output=logging/%x_%j.out  
#SBATCH --error=logging/%x_%j.err

export STR_TYPE='smiles'
export SUBSET='aspirin'
export SUBSET_LENGTH=8284280 

export VALID_LEN=10000
export GEN_LEN=1000000
export GEN_LEN_STR='1M'

export TOP_K=192
export TOP_P=1.0
export TEMPERATURE=1.0

export GEN_TYPE="top_k_${TOP_K}_top_p_${TOP_P}_temperature_${TEMPERATURE}"
export PRETRAIN='rand'
export FINETUNE='rand'
export PROBS="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/ablations/perplexities/correct_validations/${STR_TYPE}/valid_${PRETRAIN}_${FINETUNE}/${SUBSET}/top_k_${TOP_K}_top_p_${TOP_P}_temperature_${TEMPERATURE}.csv"
export ACTUAL_STATS="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/statistics/actual_generation_statistics/${STR_TYPE}/${SUBSET}/${GEN_TYPE}_gen_len_${GEN_LEN_STR}_${PRETRAIN}_${FINETUNE}.xlsx"
export OUT_PATH="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/statistics/predicted_generation_statistics/${STR_TYPE}/${SUBSET}/${GEN_TYPE}_gen_len_${GEN_LEN_STR}_${PRETRAIN}_${FINETUNE}_valid_len_${VALID_LEN}_temp.xlsx" 

python /nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/utils/metrics_modeling.py \
    --subset $SUBSET \
    --subset_length $SUBSET_LENGTH \
    --valid_probs_csv $PROBS \
    --valid_length $VALID_LEN \
    --gen_actual_xlsx $ACTUAL_STATS\
    --gen_length $GEN_LEN \
    --out_path $OUT_PATH