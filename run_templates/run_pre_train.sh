#!/bin/bash

#SBATCH --job-name=MolgenTrain
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:8
#SBATCH --mem=100gb
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err

export LR="4.00E-04"
export BS="128"
export MODEL="OPT_1.2B_ep_1_all_rand_sf_848M"
export GRAD_ACC="32"


accelerate launch --config_file ../accelerate_fsdp_config.yaml \
     ../src/train_with_trainer.py \
    --seed 1 \
    --output_dir ../src/checkpoints/$MODEL"_"$LR"_"hf_gradacc_$GRAD_ACC \
    --dataset_name ../src/data/data/data_bin_all_rand_sf_848M \
    --tokenizer_name ../src/data/tokenizers/tokenizer_sf/tokenizer.json \
    --resume_from_checkpoint "" \
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
    --gradient_accumulation_steps $GRAD_ACC \
    --preprocessing_num_workers 30 \
    --logging_steps 1 \
    --eval_steps 100 \
    --max_steps -1 \
    --save_steps 250 \
    --warmup_steps 2588 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --report_to none \
    --save_safetensors False \
    --aim_exp_name "$MODEL"_"$LR"_"bs_8x$GRAD_ACC"x"$BS"_"hf, lr=$LR." \
    --aim_repo_dir "../" \
    --train_with_sample_size 0 \
    --gradient_checkpointing False