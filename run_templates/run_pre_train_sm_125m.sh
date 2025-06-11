export PRE_TRAIN="canon"
export MOL_REPR="sm"
export VOCAB_SIZE=584 # 192/584

export LR="4.00E-04"
export BS="512"
export GRAD_ACC="64"
export LOSS_TYPE="mean"


accelerate launch --config_file ../accelerate_fsdp_config_1.yaml \
     ../src/train_with_molecular_batch.py \
    --seed 1 \
    --output_dir ../src/checkpoints/pre_trained/OPT_125m_ep_1_all_$PRE_TRAIN"_"$MOL_REPR"_"131M_test \
    --dataset_name ../src/data/data/data_bin_all_$PRE_TRAIN"_"$MOL_REPR"_131M" \
    --tokenizer_name ../src/data/tokenizers/tokenizer_$MOL_REPR/tokenizer.json \
    --resume_from_checkpoint "" \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --config_name facebook/opt-125m \
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
    --preprocessing_num_workers 20 \
    --logging_steps 1 \
    --eval_steps 500 \
    --max_steps -1 \
    --save_steps 1000 \
    --warmup_steps 2588 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --report_to none \
    --save_safetensors False \
    --aim_exp_name "OPT_125m_ep_1_all_$PRE_TRAIN"_"$MOL_REPR"_"131M, lr=$LR, pre-train 125m model on SMILES, test the dataloader." \
    --aim_repo_dir "../" \
    --train_with_sample_size 0 \
    --gradient_checkpointing False \
    --save_total_limit 1 \
    --loss_type $LOSS_TYPE \
    --shuffle_train True