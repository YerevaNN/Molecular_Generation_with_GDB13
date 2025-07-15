#!/bin/bash

#SBATCH --job-name=CalcStats_100K
#SBATCH --cpus-per-task=20       
#SBATCH --mem=20gb
#SBATCH --time=100:00:00                        
#SBATCH --output=logging/%x_%j.out  
#SBATCH --error=logging/%x_%j.err
#SBATCH --partition=all


export SUBSET_PATH="/nfs/dgx/raid/molgen/generations/code/data_level_3.2_depth_8_3_vars_canon_forms.txt"
export GEN_PATH="../src/ablations/generations/generations/code/model_ep_4_gen_100K_temp_0.5.txt"
export BYTECODES_PATH="/nfs/dgx/raid/molgen/generations/code/data_level_3.2_depth_8_3_vars_bytecodes.txt"
export OUT_PATH="../src/ablations/statistics/actual_generation_statistics/code/Llama3.2_1B_ep_4_gen_1M_temp_1_beam.xlsx"
# export get_first=400000

                                                                                                                
python ../src/utils/get_code_gen_set_intersection.py \
    --subset_path $SUBSET_PATH \
    --bytecodes_path $BYTECODES_PATH \
    --gen_path $GEN_PATH \
    --output_path $OUT_PATH \
    # --get_first $get_first \