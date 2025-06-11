#!/bin/bash

#SBATCH --job-name=Beam10MAspirin
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=70gb
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err


export MOL_REPR="sf"
export VOCAB_SIZE=192 # 192 / 584
export SUBSET="aspirin_0.4"
export LR="8.00E-05"
export MODEL_NAME="OPT_1.2B_ep_1_all_canon_finetune_all_canon_${SUBSET}_${MOL_REPR}_1000K_${LR}" 
export ITERATION=3900

export ITER_LEN=33
export TEMPERATURE=1.0

export N=10
export GEN_LEN=$((N * 1000000))
export GEN_LEN_STR=$N"M" 

export BATCH_SIZE=8196

export TOKENIZER="../src/data/tokenizers/tokenizer_${MOL_REPR}/tokenizer.json"
export MODEL="../src/checkpoints/fine_tuned/${MOL_REPR}/$MODEL_NAME/checkpoint-${ITERATION}"
export OUT_PATH="../src/ablations/generations/generations/beam_search/${MODEL_NAME}_beam_${GEN_LEN_STR}.csv"


accelerate launch --config_file ../accelerate_fsdp_config_3.yaml \
     ../src/beam_search.py \
     --tokenizer_path $TOKENIZER \
     --vocab_size $VOCAB_SIZE \
     --resume_from_checkpoint $MODEL \
     --output_path $OUT_PATH \
     --temperature $TEMPERATURE \
     --iter_len $ITER_LEN \
     --gen_len $GEN_LEN \
     --batch_size $BATCH_SIZE 