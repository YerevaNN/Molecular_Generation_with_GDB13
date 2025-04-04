#!/bin/bash

#SBATCH --job-name=TPTP
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:0
#SBATCH --mem=30gb
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err


# took about 5-6 hours 
python ../src/data/ACE_data/to_tptp.py \
    --input /nfs/h100/raid/chem/all_sentences_17.txt \
    --output /nfs/h100/raid/chem/all_sentences_17.tptp \
    --ape ../src/data/ACE_data/APE/ape.exe \
    --workers 60