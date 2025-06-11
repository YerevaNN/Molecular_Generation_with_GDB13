#!/bin/bash

#SBATCH --job-name=TPTP_17_passive
#SBATCH --cpus-per-task=100
#SBATCH --gres=gpu:0
#SBATCH --mem=30gb
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/vampire/%x_%j.out
#SBATCH --error=logging/vampire/%x_%j.err

#SBATCH --partition=all


# took about 5-6 hours 
python ../src/data/ACE_data/to_tptp.py \
    --input /nfs/h100/raid/chem/all_sentences_17_passive_fixed.txt \
    --output /nfs/h100/raid/chem/all_sentences_17_passive_fixed.tptp \
    --ape ../src/data/ACE_data/APE/ape.exe \
    --workers 100