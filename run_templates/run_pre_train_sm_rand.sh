#!/bin/bash

#SBATCH --job-name=SmRandPretrain
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:4
#SBATCH --mem=60gb
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err

export PRE_TRAIN="rand"
export MOL_REPR="sm"
export VOCAB_SIZE=584 # 192/584

export LR="1.00E-04"
export BS="128"
export GRAD_ACC="64"
export LOSS_TYPE="mean"


accelerate launch --config_file ../accelerate_fsdp_config_4gpu.yaml \
     ../src/train_with_molecular_batch.py \
    --seed 1 \
    --output_dir ../src/checkpoints/pre_trained/OPT_1.2B_ep_1_all_$PRE_TRAIN"_"$MOL_REPR"_"848M_lr_1e-4 \
    --dataset_name ../src/data/data/data_bin_all_$PRE_TRAIN"_"$MOL_REPR"_848M" \
    --tokenizer_name ../src/data/tokenizers/tokenizer_$MOL_REPR/tokenizer.json \
    --resume_from_checkpoint "../src/checkpoints/pre_trained/OPT_1.2B_ep_1_all_$PRE_TRAIN"_"$MOL_REPR"_"848M_lr_1e-4/checkpoint-25000" \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --config_name facebook/opt-1.3B \
    --max_position_embeddings 64 \
    --dropout 0.0 \
    --vocab_size $VOCAB_SIZE \
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
    --eval_steps 250 \
    --max_steps -1 \
    --save_steps 250 \
    --warmup_steps 2588 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --report_to none \
    --save_safetensors False \
    --aim_exp_name "OPT_1.2B_ep_1_all_$PRE_TRAIN"_"$MOL_REPR"_"848M, lr=$LR, pre-train 1.2B model on rand SMILES, continue." \
    --aim_repo_dir "../" \
    --train_with_sample_size 0 \
    --gradient_checkpointing False \
    --loss_type $LOSS_TYPE \
    --shuffle_train True