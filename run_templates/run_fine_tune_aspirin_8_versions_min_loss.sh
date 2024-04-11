#!/bin/bash

#SBATCH --job-name=Aspirin8VersionMinLoss
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=40gb
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --output=logging/%x_%j.out
#SBATCH --error=logging/%x_%j.err


export PRE_TRAIN="canon"
export MOL_REPR="sf"
export VOCAB_SIZE=192 # 192/600
export DATA_SPLIT="all_rand_aspirin_0.4" 
export DATA_SUF="_rand_8_versions" # _rand_8_versions
export LR=""
export GRAD_ACC=1
export BS_TRAIN=256
export BS_VALID=128
export WARMUP=391


for LR in "8.00E-05" "16.00E-05" 
do
accelerate launch --config_file ../accelerate_fsdp_config_2.yaml \
     ../src/train_with_molecular_batch.py \
    --resume_from_checkpoint "" \
    --finetune_from_checkpoint "../src/checkpoints/pre_trained/OPT_1.2B_ep_1_all_$PRE_TRAIN"_"$MOL_REPR"_"848M/checkpoint-25750/pytorch_model.bin" \
    --dataset_name ../src/data/data/data_bin_$DATA_SPLIT"_"$MOL_REPR"_"1000K$DATA_SUF \
    --tokenizer_name ../src/data/tokenizers/tokenizer_$MOL_REPR/tokenizer.json \
    --output_dir ../src/checkpoints/fine_tuned/OPT_1.2B_ep_1_all_$PRE_TRAIN"_"finetune_$DATA_SPLIT"_"$MOL_REPR"_"1000K$DATA_SUF"_"$LR"_"min_loss \
    --aim_exp_name "OPT_1.2B_ep_1_all_$PRE_TRAIN"_"finetune_$DATA_SPLIT"_"$MOL_REPR"_"1000K$DATA_SUF"_"$LR, valid_data=0.5K all versions, sum_of_probs, min loss." \
    --seed 1 \
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
    --dataloader_pin_memory \
    --dataloader_drop_last \
    --preprocessing_num_workers 20 \
    --gradient_accumulation_steps $GRAD_ACC \
    --fp16 \
    --fp16_backend auto \
    --half_precision_backend auto \
    --report_to none \
    --save_safetensors False \
    --aim_repo_dir "../" \
    --train_with_sample_size 0 \
    --gradient_checkpointing False \
    --save_total_limit 1 \
    --local_rank 0 \
    --log_on_each_node \
    --logging_steps 1 \
    --eval_steps 100 \
    --max_steps -1 \
    --save_steps 1000 \
    --warmup_steps $WARMUP \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BS_TRAIN \
    --per_device_eval_batch_size $BS_VALID \
    --learning_rate $LR \
    --train_with_min_loss_seq True
done    