#!/bin/bash

#SBATCH --job-name=BeamSearch
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err

export SUBSET='aspirin'
export EXT='aspirin_0.4'
export REPR='selfies'
export SHORT_REPR='sf'

export PRETRAIN='canon'
export FINETUNE='rand'
export LR='8.00E-05'
export ITERATION=3900

export ITER_LEN=33
export TEMPERATURE=1.0
export GEN_LEN_STR='1M'

export MODEL="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/checkpoints/main_checkpoints/OPT_1.2B_ep_1_all_${PRETRAIN}_finetune_all_${FINETUNE}_${EXT}_${SHORT_REPR}_1000K_${LR}_hf/checkpoint-${ITERATION}"
export TOKENIZER="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/data/tokenizers/tokenizer_${SHORT_REPR}/tokenizer.json"
export OUT_PATH="/nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/generations/${REPR}/${SUBSET}_Beam_${GEN_LEN_STR}_temp_${TEMPERATURE}_${PRETRAIN}_${FINETUNE}_iter_length_${ITER_LEN}.csv"

python /nfs/c9/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/utils/beam_search.py \
     --tokenizer_path $TOKENIZER \
     --resume_from_checkpoint $MODEL \
     --output_beams $OUT_PATH \
     --temperature $TEMPERATURE \
     --iter_len $ITER_LEN \
     --gen_len 1000000 \
     --batch_size 4096 

