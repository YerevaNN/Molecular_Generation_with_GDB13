#!/bin/bash

#SBATCH --job-name=FineTune
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err



export DATA_SPLIT="aspirin_0.4" 
export LR="2.00E-05"

accelerate launch --config_file ../accelerate_fsdp_config.yaml \
     ../src/train_with_trainer.py \
    --seed 1 \
    --output_dir ../src/checkpoints/OPT_1.2B_ep_1_$DATA_SPLIT"_"sf_1000K_$LR"_"hf \
    --dataset_name ../src/data/data/data_bin_$DATA_SPLIT"_"sf_1000K \
    --tokenizer_name ../src/data/tokenizers/tokenizer_sf/tokenizer.json \
    --fine_tune_from "../src/checkpoints/OPT_1.2B_ep_1_all_rand_sf_848M_4.00E-04_hf_gradacc_32/checkpoint-25750/pytorch_model.bin" \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --config_name facebook/opt-1.3B \
    --max_position_embeddings 64 \
    --dropout 0.0 \
    --vocab_size 192 \
    --optim adamw_torch \
    --adam_beta2 0.95 \
    --weight_decay 0.1 \
    --dataloader_num_workers 1 \
    --dataloader_drop_last \
    --dataloader_pin_memory \
    --fp16 \
    --fp16_backend auto \
    --half_precision_backend auto \
    --learning_rate $LR \
    --local_rank 0 \
    --log_on_each_node \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 30 \
    --logging_steps 1 \
    --eval_steps 100 \
    --max_steps -1 \
    --save_steps  300 \
    --warmup_steps 391 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --report_to none \
    --save_safetensors False \
    --aim_exp_name "OPT_1.2B_ep_1_$DATA_SPLIT"_"sf_1000K_$LR"_"bs_1x1x256_hf, lr=$LR." \
    --aim_repo_dir "../" \
    --train_with_sample_size 0 \
    --gradient_checkpointing False \
    --load_best_model_at_end True \
    --save_total_limit 3