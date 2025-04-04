#!/bin/bash

#SBATCH --job-name=TPTP_group
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=30gb
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err



python ../src/data/ACE_data/group.py \
    --text /nfs/h100/raid/chem/all_sentences_17.txt \
    --tptp /nfs/h100/raid/chem/all_sentences_17.tptp \
    --output /nfs/h100/raid/chem/cf_17