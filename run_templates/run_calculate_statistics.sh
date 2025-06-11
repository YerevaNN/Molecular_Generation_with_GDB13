#!/bin/bash

#SBATCH --job-name=CalcStats
#SBATCH --cpus-per-task=20       
#SBATCH --mem=20gb
#SBATCH --time=01:00:00                        
#SBATCH --output=logging/%x_%j.out  
#SBATCH --error=logging/%x_%j.err

export SUBSET='aspirin_0.4'  # aspirin_0.4, sas_3, druglike_0.4, equal_dist
export MOL_REPR='sf'
export SUBSET_PATH="../src/data/data/subsets/${SUBSET}_${MOL_REPR}.jsonl"

export N=10
export GEN_LEN=$((N * 1000000))
export GEN_LEN_STR=$N"M"

export GEN_PATH="../src/ablations/generations/generations/sf/predicted/OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_gen_10M_temp_1.4.csv"
export OUT_PATH="../src/ablations/statistics/actual_generation_statistics/${MOL_REPR}/OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_gen_10M_temp_1.4.xlsx"

                                                                                                                                   
python ../src/calculate_statistics.py \
    --mol_repr $MOL_REPR \
    --subset_path $SUBSET_PATH \
    --generation_path $GEN_PATH \
    --output_path $OUT_PATH \
    --gen_length $GEN_LEN \
    --chunk_size 100000 \
    --num_processes 4 