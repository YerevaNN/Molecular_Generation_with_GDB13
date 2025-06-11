#!/bin/bash

#SBATCH --job-name=GenAsp1Temp
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err


export N=1
export GEN_LEN=$((N * 1000000))
export GEN_LEN_STR=$N"M" 

export ITERATION=3900
export MOL_REPR="sf"
export VOCAB_SIZE=192
export TOP_K=192
export TOP_P=1.0
export TEMPERATURE=1.0
export MODEL="OPT_1.2B_ep_1_all_canon_finetune_all_canon_aspirin_0.4_sf_1000K_8.00E-05"

for temp in 0.6 0.8 1.5
do
     accelerate launch --config_file ../accelerate_fsdp_config_gen_opt.yaml \
          ../src/generate.py \
     --tokenizer_name ../src/data/tokenizers/tokenizer_${MOL_REPR}/tokenizer.json \
     --resume_from_checkpoint ../src/checkpoints/fine_tuned/sf/$MODEL/checkpoint-$ITERATION \
     --output_dir "../src/ablations/generations/generations/$MOL_REPR/temperatures/$MODEL"_"iter_$ITERATION"_gen_"$GEN_LEN_STR"_temp_"$temp.csv" \
     --batch_size 4096 \
     --gen_len $GEN_LEN \
     --vocab_size $VOCAB_SIZE \
     --top_k $TOP_K \
     --top_p $TOP_P \
     --temperature $temp
done     