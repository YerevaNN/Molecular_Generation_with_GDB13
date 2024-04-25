#!/bin/bash

#SBATCH --job-name=CalcStats                     
#SBATCH --cpus-per-task=20         
#SBATCH --mem=20gb                  
#SBATCH --time=01:00:00                          
#SBATCH --output=logging/%x_%j.out  
#SBATCH --error=logging/%x_%j.err

export REPR='selfies' 
export SUBSET='sas'

export PRE_TRAIN='canon'
export FINE_TUNE='canon'

export FROM_GEN='10M'
export GEN_LEN_STR='1M'
export GEN_LENGTH=1000000
export TEMPERATURE=1

export SUBSET_PATH='/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_project/Molecular_Generation_with_GDB13/data/sas_3_sf/sas_3_sf.jsonl'
export GEN_PATH="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/ablations/generations/generations/generations_10M/${SUBSET}"
export OUT_PATH="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/statistics/actual_generation_statistics/${REPR}/${SUBSET}"
                                                                                                                                       
python /nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/utils/Calculate_statistics.py \
    --repr_type $REPR \
    --subset_path $SUBSET_PATH \
    --generation_path "$GEN_PATH/Pretrain_${PRETRAIN}-Finetune_${FINETUNE}-Tempetarure_${TEMPERATURE}.csv" \
    --output_path "$OUT_PATH/PRETRAIN_${PRETRAIN}_FINETUNE_${FINETUNE}_FROM_GEN_${FROM_GEN}_GEN_LEN_${GEN_LEN_STR}_TEMP_${TEMPERATURE}.xlsx" \
    --gen_length $GEN_LENGTH \
    --chunk_size 100000 \
    --num_processes 20 