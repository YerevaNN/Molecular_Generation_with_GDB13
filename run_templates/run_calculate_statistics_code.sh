#!/bin/bash

#SBATCH --job-name=CalcStats_1M_0.8
#SBATCH --cpus-per-task=20       
#SBATCH --mem=20gb
#SBATCH --time=100:00:00                        
#SBATCH --output=logging/%x_%j.out  
#SBATCH --error=logging/%x_%j.err
#SBATCH --partition=all


export SUBSET_PATH="/nfs/dgx/raid/molgen/code_recall/data/data_level_3.2_depth_8_3_vars_canon_forms.txt"
export GEN_PATH="../src/ablations/generations/generations/code/Llama3.2_1B_ep_4_gen_1M_temp_0.8.txt"
export BYTECODES_PATH="/nfs/dgx/raid/molgen/code_recall/data/data_level_3.2_depth_8_3_vars_bytecodes.txt"
export OUT_PATH="../src/ablations/statistics/actual_generation_statistics/code/Llama3.2_1B_ep_4_gen_1M_temp_1.xlsx"
# export get_first=700000

                                                                                                                                   
python ../src/utils/get_code_gen_set_intersection.py \
    --subset_path $SUBSET_PATH \
    --bytecodes_path $BYTECODES_PATH \
    --gen_path $GEN_PATH \
    --output_path $OUT_PATH \
    # --get_first $get_first \