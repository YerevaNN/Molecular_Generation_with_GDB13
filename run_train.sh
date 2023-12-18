export CUDA_VISIBLE_DEVICES="0"
export PROJECT_NAME=Molecular_Generation_with_GDB13 

accelerate launch --config_file ./accelerate_config.yaml \
        src/train.py \
        --seed 1 \
        --output_dir ./src/checkpoints/OPT_85M_ep_1_sas_3_sf_1000K_6.00E-05_new \
        --dataset_name ./src/data/data/data_bin_sas_3_sf_1000K \
        --tokenizer_name ./src/data/tokenizers/tokenizer_sf/tokenizer.json \
        --resume_from_checkpoint "" \
        --config_name "facebook/opt-125m" \
        --vocab_size 192 \
        --max_position_embeddings 64 \
        --preprocessing_num_workers 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1\
        --num_train_epochs 1 \
        --lr_scheduler_type "linear" \
        --learning_rate 6.00E-05 \
        --num_warmup_steps 781 \
        --checkpointing_steps 100 \
        --validation_steps 100 \
        --weight_decay 0.1 \
        --with_tracking \
        --log_with "aim" \
        --track_dir ./ \
         # --experiment_description "OPT_85M_ep_1_sas_3_sf_1000K_6.00E-05 pre-train" \
        #  --resume_from_checkpoint "./src/checkpoints/OPT_85M_ep_1_sas_3_sf_1000K_6.00E-05/checkpoint_last.pt" \
        
