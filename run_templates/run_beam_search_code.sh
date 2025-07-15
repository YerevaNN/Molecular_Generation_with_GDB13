#!/bin/bash

#SBATCH --job-name=LocBeam700K_dgx
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=70gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err
#SBATCH --partition=a100

# export CUDA_VISIBLE_DEVICES=1

export ITER_LEN=45
export TEMPERATURE=1.0

export N=7
export GEN_LEN=$((N * 100000))
export GEN_LEN_STR=$N"M" 

export BATCH_SIZE=128 # ???

export TOKENIZER="meta-llama/Llama-3.2-1B"
export MODEL="/auto/home/knarik/Molecular_Generation_with_GDB13/src/checkpoints/checkpoints_code/Llama-3-1B_tit_hf_4_epochs/step-3126"
export OUT_PATH="../src/ablations/generations/generations/code/model_ep_4_gen_100K_beam_700K_cover_loc_dgx.csv"
export VOCAB_PATH="../src/ablations/generations/generations/code/train_vocab.txt"


accelerate launch --config_file ../accelerate_local_config.yaml \
     ../src/beam_search_code.py \
     --vocab_path $VOCAB_PATH \
     --tokenizer_path $TOKENIZER \
     --resume_from_checkpoint $MODEL \
     --output_path $OUT_PATH \
     --temperature $TEMPERATURE \
     --iter_len $ITER_LEN \
     --gen_len $GEN_LEN \
     --batch_size $BATCH_SIZE 