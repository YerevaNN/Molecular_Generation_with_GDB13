#!/bin/bash

#SBATCH --job-name=PredMetrics                     
#SBATCH --cpus-per-task=1        
#SBATCH --mem=20gb                  
#SBATCH --time=02:00:00                          
#SBATCH --output=logging/%x_%j.out  
#SBATCH --error=logging/%x_%j.err

export STR_TYPE='selfies'
export SUBSET='aspirin_0.4'
export SUBSET_LENGTH=8284280    # 8284280, 6645440, 9331077, 8051185 

export VALID_LEN=10000
export GEN_LEN=1000000
export GEN_LEN_STR='1M'
export TEMPERATURE=1.4

export PRETRAIN='canon'
export FINETUNE='canon'
export PROBS="../src/ablations/perplexities/sf/temps/valid_OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_givenTemp_$TEMPERATURE.csv"
export ACTUAL_STATS="../src/ablations/statistics/actual_generation_statistics/sf/OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_gen_1M_temp_1.xlsx"
export OUT_PATH="../src/ablations/statistics/predicted_generation_statistics/sf/OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05_iter_3900_gen_$GEN_LEN_STR"_"temp_$TEMPERATURE"_"all_gens.csv" 

python ../src/utils/metrics_modeling.py \
    --subset $SUBSET \
    --subset_length $SUBSET_LENGTH \
    --valid_probs_csv $PROBS \
    --valid_length $VALID_LEN \
    --gen_actual_xlsx $ACTUAL_STATS\
    --gen_length $GEN_LEN \
    --out_path $OUT_PATH