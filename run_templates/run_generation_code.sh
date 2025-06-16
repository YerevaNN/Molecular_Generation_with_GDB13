#!/bin/bash

#SBATCH --job-name=GenCode
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err
#SBATCH --partition=all


export CUDA_VISIBLE_DEVICES=1

export N=1
export GEN_LEN=$((N * 1000000))
export GEN_LEN_STR=$N"M" 
export VOCAB_SIZE=192
export TOP_K=192
export TOP_P=1.0
export TEMPERATURE=1.0
export MODEL="LLama"

for temp in 0.8
do
     accelerate launch --config_file ../accelerate_fsdp_config_gen_llama.yaml \
          ../src/generate_code.py \
     --tokenizer_name "meta-llama/Llama-3.2-1B" \
     --resume_from_checkpoint /nfs/dgx/raid/molgen/code_recall/checkpoints/Llama-3-1B_tit_hf_4_epochs/step-3126/ \
     --output_dir "../src/ablations/generations/generations/code/model_ep_4_gen_"$GEN_LEN_STR"_temp_"$temp.txt\
     --batch_size 5000 \
     --gen_len $GEN_LEN \
     --vocab_size $VOCAB_SIZE \
     --top_k $TOP_K \
     --top_p $TOP_P \
     --temperature $temp
done