export PROJECT_NAME=Molecular_Generation_with_GDB13 

accelerate launch --config_file ./accelerate_config.yaml \
        src/train.py \
        --seed 1 \
        --output_dir ./src/checkpoints/OPT_85M_ep_1_sas_3_sf_1000K_6.00E-05_new \
        --dataset_name ./src/data/data/data_bin_sas_3_sf_1000K \
        --tokenizer_name ./src/data/tokenizers/tokenizer_sf/tokenizer.json \
        --resume_from_checkpoint "./src/checkpoints/OPT_85M_ep_1_sas_3_sf_1000K_6.00E-05_new" \
        --config_name "facebook/opt-125m" \
        --vocab_size 192 \
        --max_position_embeddings 64 \
        --preprocessing_num_workers 1 \
        --per_device_train_batch_size 256 \
        --per_device_eval_batch_size 256 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 1 \
        --lr_scheduler_type "linear" \
        --learning_rate 6.00E-05 \
        --num_warmup_steps 10 \
        --checkpointing_steps 100 \
        --validation_steps 50 \
        --weight_decay 0.1 \
        --with_tracking \
        --track_dir ./ \
        --experiment_name "OPT_85M_ep_1_sas_3_sf_1000K_6.00E-05 resume" \
        --aim_resume_hash "4a9ac8416b6c4c07b294f111"
        # --resume_from_checkpoint ./src/checkpoints/OPT_85M_ep_1_sas_3_sf_1000K_6.00E-05_new/checkpoint_300
        # --experiment_description "OPT_85M_ep_1_sas_3_sf_1000K_6.00E-05 pre-train" \
        #  --resume_from_checkpoint "./src/checkpoints/OPT_85M_ep_1_sas_3_sf_1000K_6.00E-05/checkpoint_last.pt" \
        
