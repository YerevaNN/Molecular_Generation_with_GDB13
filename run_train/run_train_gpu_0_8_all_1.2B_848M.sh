CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" metaseq-train --task streaming_language_modeling ../Molecular_Generation_with_GDB13/data/data_bin_half_rand_sf_848M/ --sample-break-mode eos_pad_8 --hf-tokenizer ../Molecular_Generation_with_GDB13/data/tokenizers/tokenizer_sf/tokenizer.json --train-subset train --valid-subset valid --combine-valid-subsets --no-reshard-after-forward --use-sharded-state --checkpoint-activations --full-megatron-init --megatron-init-sigma 0.006 --activation-fn relu --arch transformer_lm --share-decoder-input-output-embed --decoder-layers 24 --decoder-embed-dim 2048 --decoder-ffn-embed-dim 8192 --decoder-attention-heads 32 --decoder-learned-pos --no-scale-embedding --dropout 0.0 --attention-dropout 0.0 --no-emb-dropout --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.95)" --adam-eps 1e-08 --clip-norm 1.0 --clip-norm-type l2 --criterion cross_entropy --required-batch-size-multiple 1 --distributed-world-size 8 --model-parallel-size 1 --ddp-backend fully_sharded --memory-efficient-fp16 --fp16-init-scale 4 --fp16 --seed 1 --num-workers 0 --num-workers-valid 0 --lr-scheduler polynomial_decay --lr 2.00E-04 --end-learning-rate 2.00E-05 --warmup-updates 10352 --total-num-update 103516 --max-update 103516 --tokens-per-sample 64 --batch-size 1024 --update-freq 1 --log-format json --log-interval 1 --ignore-unused-valid-subsets --validate-interval-updates 500 --wandb-project Scaling_Laws --wandb-run-name OPT_1.2B_ep_1_half_rand_sf_848M_2.00E-04_bs_8x1024 --save-interval-updates 5000 --keep-last-updates 1 --save-dir ./checkpoints/OPT_1.2B_ep_1_half_sf_848M_2.00E-04 --restore-file ""