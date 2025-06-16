#!/bin/bash

#SBATCH --job-name=PredMetrics                     
#SBATCH --cpus-per-task=1        
#SBATCH --mem=20gb                  
#SBATCH --time=02:00:00                          
#SBATCH --output=logging/%x_%j.out  
#SBATCH --error=logging/%x_%j.err

export SUBSET_LENGTH=656656 

export VALID_LEN=10000
export GEN_LEN=1000000
export GEN_LEN_STR='1M'
export TEMPERATURE=1.0

export PROBS="/auto/home/knarik/Molecular_Generation_with_GDB13/src/ablations/perplexities/code/Llama3.2_1B_ep_4.csv"
# export ACTUAL_STATS="../src/ablations/statistics/actual_generation_statistics/sf/OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_gen_1M_temp_1.xlsx"
# export OUT_PATH="../src/ablations/statistics/predicted_generation_statistics/sf/OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_gen_$GEN_LEN_STR"_"temp_$TEMPERATURE"_"all_gens.csv" 

python ../src/utils/predict_metrics.py \
    --subset_length $SUBSET_LENGTH \
    --valid_probs_csv $PROBS \
    --valid_length $VALID_LEN \
    --gen_length $GEN_LEN 