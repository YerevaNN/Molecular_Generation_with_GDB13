#!/bin/bash

#SBATCH --job-name=Vampire_336528_1_final_level_5_new_parallel_processpool_optim_multitask
#SBATCH --cpus-per-task=100
#SBATCH --gres=gpu:0
#SBATCH --mem=10gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/vampire/%x_%j.out
#SBATCH --error=logging/vampire/%x_%j.err

#SBATCH --partition=h100

## /cf_17/clusters/big-card-customer-inserts-small/clusters_level_1

python ../src/data/ACE_data/theorem_prover_linear_with_clusters_optim.py \
    --input_path /nfs/h100/raid/chem/cf_17/clusters_new/big-card-customer-inserts-small/clusters_level_4 \
    --output_path /nfs/h100/raid/chem/cf_17/clusters_new/big-card-customer-inserts-small/clusters_level_5_parallel_processpool_2_optim_multitask \
    --chunk_size 2 \
    --num_workers 120 \
    --num_threads 50 \