#!/bin/bash

#SBATCH --job-name=Vampire_1269492_level_2
#SBATCH --cpus-per-task=100
#SBATCH --gres=gpu:0
#SBATCH --mem=10gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/vampire/%x_%j.out
#SBATCH --error=logging/vampire/%x_%j.err

#SBATCH --partition=all

## /cf_17_new/clusters/big-card-customer-inserts-takes/clusters_level_1

python ../src/data/ACE_data/theorem_prover_linear_with_clusters.py \
    --input_path /nfs/h100/raid/chem/cf_17_new/clusters/big-card-customer-inserts-takes/clusters_level_1 \
    --output_path /nfs/h100/raid/chem/cf_17_new/clusters/big-card-customer-inserts-takes/clusters_level_2 \
    --chunk_size 2 \
    --num_workers 120 \
    --num_threads 10 \