#!/bin/bash

#SBATCH --job-name=Vampire_336528_1_level_1
#SBATCH --cpus-per-task=100
#SBATCH --gres=gpu:0
#SBATCH --mem=10gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/vampire/%x_%j.out
#SBATCH --error=logging/vampire/%x_%j.err

#SBATCH --partition=all

## /cf_17/clusters/big-card-customer-inserts-small/clusters_level_1

python ../src/data/ACE_data/theorem_prover_linear_with_clusters.py \
    --input_path /nfs/h100/raid/chem/cf_17/big-card-customer-inserts-small.tptp \
    --output_path /nfs/h100/raid/chem/cf_17/clusters/big-card-customer-inserts-small \
    --chunk_size 1000 \
    --num_workers 120 \
    --num_threads 2 \