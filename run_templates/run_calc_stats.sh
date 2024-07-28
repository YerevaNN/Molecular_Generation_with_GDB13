#!/bin/bash

#SBATCH --job-name=CalcStats                     
#SBATCH --cpus-per-task=20         
#SBATCH --mem=20gb                  
#SBATCH --time=01:00:00                          
#SBATCH --output=logging/%x_%j.out  
#SBATCH --error=logging/%x_%j.err

# export SUBSET='aspirin'
# export EXT='aspirin_0.4'
# export LR='8.00E-05'

# export SUBSET='sas'
# export EXT='sas_3'
# export LR='8.00E-05'

# export SUBSET='druglike'
# export EXT='druglike_0.4'
# export LR='8.00E-05'

export SUBSET='eqdist'
export SUBSET_EXT='equal_dist'

export STR_TYPE='selfies' 
export SHORT_STR_TYPE='sf'

export PRETRAIN='rand'
export FINETUNE='rand'
export ITERATION=3900
export LR='16.00E-05'

export GEN_LEN=1000000
export GEN_LEN_STR='1M'
export GEN_TYPE="top_k_${TOP_K}_top_p_${TOP_P}_temperature_${TEMPERATURE}"

export SUBSET_PATH="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_project/Molecular_Generation_with_GDB13/data/${SUBSET}/${STR_TYPE}/${SUBSET_EXT}_${SHORT_STR_TYPE}.jsonl"
export GEN_PATH="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/ablations/generations/generations/${SHORT_STR_TYPE}/OPT_1.2B_ep_1_all_${PRETRAIN}_finetune_all_${FINETUNE}_${SUBSET_EXT}_${SHORT_STR_TYPE}_1000K_${LR}_hf_iter_${ITERATION}_gen_${GEN_LEN}__.csv" 
export OUT_PATH="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/statistics/actual_generation_statistics/${STR_TYPE}/${SUBSET}/${GEN_TYPE}_gen_len_${GEN_LEN_STR}_${PRETRAIN}_${FINETUNE}.xlsx"
                                                                                                                                   
python /nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/utils/calculate_statistics.py \
    --str_type $STR_TYPE \
    --subset_path $SUBSET_PATH \
    --generation_path $GEN_PATH \
    --output_path $OUT_PATH \
    --gen_length $GEN_LEN \
    --chunk_size 10000 \
    --num_processes 4 