export CUDA_VISIBLE_DEVICES="1"
export PROJECT_NAME=Molecular_Generation_with_GDB13 
export DATA_SIZE="64K"
export BATCH_SIZE=128
export STEPS=5000
export WARMUP=500
export EPOCH=10
export LR=1.8e-4


for dropout in 0.5 0.05
do
    END_LR=$(awk "BEGIN { printf \"%.7f\", $LR / 10 }")

    metaseq-train --task streaming_language_modeling \
                    ../$PROJECT_NAME/data/data_bin_aspirin_0.4_sf_$DATA_SIZE/ \
                    --sample-break-mode "eos_pad_8" \
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
                    --decoder-layers 24 \
                    --decoder-embed-dim 1536 \
                    --decoder-ffn-embed-dim 6144 \
                    --decoder-attention-heads 16 \
                    --decoder-learned-pos \
                    --no-scale-embedding \
                    --dropout $dropout \
                    --attention-dropout $dropout \
                    --no-emb-dropout \
                    --optimizer adam \
                    --adam-betas  "(0.9, 0.95)" \
                    --adam-eps 1e-08 \
                    --weight-decay 0.1 \
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
                    --lr $LR \
                    --end-learning-rate $END_LR \
                    --warmup-updates $WARMUP \
                    --total-num-update $STEPS \
                    --max-update $STEPS \
                    --tokens-per-sample 64 \
                    --batch-size $BATCH_SIZE \
                    --update-freq 1 \
                    --log-format json \
                    --log-interval 1 \
                    --ignore-unused-valid-subsets \
                    --validate-interval-updates 500 \
                    --wandb-project Scaling_Laws \
                    --wandb-run-name OPT_680M_ep_$EPOCH"_aspirin_0.4_sf_"$DATA_SIZE"_1.80E-04_drop_"$dropout \
                    --save-interval-epochs $EPOCH \
                    --keep-last-updates 1 \
                    --save-dir ./checkpoints/OPT_680M_ep_$EPOCH"_aspirin_0.4_sf_"$DATA_SIZE"_1.80E-04_drop_"$dropout \
                    --restore-file "" 
done  

