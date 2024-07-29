#!/bin/bash

#SBATCH --job-name=MolgenGen
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err


export ITERATION=3900
export GEN_LEN=1000000
export PROMPT="_" # [Canon] / [Rand]
export MOL_REPR="sm"
export VOCAB_SIZE=584
export TOP_K=584
export TOP_P=1.0
export TEMPERATURE=1.0


accelerate launch --config_file ../accelerate_gen_config.yaml \
     ../src/generate.py \
    --tokenizer_name ../src/data/tokenizers/tokenizer_sf/tokenizer.json \
    --resume_from_checkpoint ../src/checkpoints/$MODEL/checkpoint-$ITERATION \
    --output_dir ../src/ablations/generations/$MODEL"_"iter_$ITERATION"_gen_"$GEN_LEN"_"$PROMPT.csv \
    --batch_size 128 \
    --prompt_token $PROMPT \
    --gen_len $GEN_LEN \
    --vocab_size $VOCAB_SIZE \
    --top_k $TOP_K \
    --top_p $TOP_P \
    --temperature $TEMPERATURE \