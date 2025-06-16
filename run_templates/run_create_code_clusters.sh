#!/bin/bash

#SBATCH --job-name=CodeClustering
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:0
#SBATCH --mem=70gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err
#SBATCH --partition=all

export BYTECODES="/nfs/dgx/raid/molgen/data/data_code/data_level_3.2_depth_8_3_vars_bytecodes.txt"
export SUBSET_PATH="/nfs/dgx/raid/molgen/data/data_code/data_level_3.2_depth_8_3_vars.txt"
export OUT_PATH="/auto/home/knarik/Molecular_Generation_with_GDB13/src/ablations/generations/generations/code/clusters.json"
 
python ../src/utils/create_equivalent_code_clusters.py \
     --subset_path $SUBSET_PATH \
     --bytecodes_path $BYTECODES \
     --output_path $OUT_PATH  