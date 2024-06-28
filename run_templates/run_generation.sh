#!/bin/bash

#SBATCH --job-name=Gen1MRand
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err


export ITERATION=3900
export GEN_LEN=1000000
export PROMPT="_" # [Canon] / [Rand]
export MOL_REPR="sm"
export VOCAB_SIZE=584

for MODEL in "OPT_1.2B_ep_1_all_rand_finetune_all_rand_aspirin_0.4_sm_1000K_4.00E-05" "OPT_1.2B_ep_1_all_rand_finetune_all_rand_druglike_0.4_sm_1000K_4.00E-05" "OPT_1.2B_ep_1_all_rand_finetune_all_rand_equal_dist_sm_1000K_4.00E-05"
do
accelerate launch --config_file ../accelerate_gen_config.yaml \
     ../src/generate.py \
    --tokenizer_name ../src/data/tokenizers/tokenizer_$MOL_REPR/tokenizer.json \
    --resume_from_checkpoint ../src/checkpoints/fine_tuned/$MODEL/checkpoint-$ITERATION \
     --output_dir ../src/ablations/generations/generations/$MOL_REPR/$MODEL"_"iter_$ITERATION"_gen_"$GEN_LEN"_"$PROMPT.csv \
    --batch_size 4096 \
    --prompt_token $PROMPT \
    --gen_len $GEN_LEN \
    --vocab_size $VOCAB_SIZE
done