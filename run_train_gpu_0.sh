export CUDA_VISIBLE_DEVICES="0"
export CUDA_LAUNCH_BLOCKING=1

export PROJECT_NAME=Molecular_Generation_with_GDB13 

metaseq-train --task streaming_language_modeling \
                ../$PROJECT_NAME/data/data_bin_0/ \
                --sample-break-mode "complete" \
                --hf-tokenizer ../$PROJECT_NAME/data/tokenizers/tokenizer_sf/tokenizer.json \
                --train-subset train \
                --valid-subset valid \
                --combine-valid-subsets \
                --no-reshard-after-forward \
                --use-sharded-state \
                --checkpoint-activations \
                --full-megatron-init \
                --megatron-init-sigma 0.006 \
                --activation-fn relu \
                --arch transformer_lm \
                --share-decoder-input-output-embed \
                --decoder-layers 4 \
                --decoder-embed-dim 128 \
                --decoder-ffn-embed-dim 512 \
                --decoder-attention-heads 2 \
                --decoder-learned-pos \
                --no-scale-embedding \
                --dropout 0.1 \
                --attention-dropout 0.1 \
                --no-emb-dropout \
                --weight-decay 0.1 \
                --optimizer adam \
                --adam-betas  "(0.9, 0.95)" \
                --adam-eps 1e-08 \
                --clip-norm 1.0 \
                --clip-norm-type l2 \
                --criterion cross_entropy \
                --required-batch-size-multiple 1 \
                --distributed-world-size 1 \
                --model-parallel-size 1 \
                --ddp-backend pytorch_ddp \
                --memory-efficient-fp16 \
                --fp16-init-scale 4 \
                --fp16 \
                --seed 1 \
                --num-workers 0 \
                --num-workers-valid 0 \
                --lr-scheduler polynomial_decay \
                --lr 0.0001 \
                --end-learning-rate 0.00001 \
                --warmup-updates 3000 \
                --total-num-update 180000 \
                --max-update 180000 \
                --tokens-per-sample 64 \
                --batch-size 1024 \
                --update-freq 1 \
                --log-format json \
                --log-interval 1 \
                --ignore-unused-valid-subsets \
                --validate-interval-updates 976 \
                --wandb-project Molecular_Generation_with_GDB13 \
                --wandb-run-name aspirin_0.4_sf \
                --save-interval-updates 1000 \
                --save-dir "./checkpoints/aspirin_0.4_sf" \
                --restore-file "./checkpoints/aspirin_0.4_sf/checkpoint_137000.pt" \