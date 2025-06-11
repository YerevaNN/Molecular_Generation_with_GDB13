#!/bin/bash

#SBATCH --job-name=TPTP_group
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=30gb
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err

# --tptp /nfs/h100/raid/chem/all_sentences_17_without_small_and_with_employee.tptp \

python ../src/data/ACE_data/group_small.py \
    --text /nfs/h100/raid/chem/all_sentences_14_passive.txt \
    --tptp /nfs/h100/raid/chem/all_sentences_14_passive.tptp \
    --output /nfs/h100/raid/chem/cf_14_passive_exp