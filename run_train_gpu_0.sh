export CUDA_VISIBLE_DEVICES="0"
export PROJECT_NAME=Molecular_Generation_with_GDB13 
# export EPOCH=80
export ONE_EPOCH_UPDATES=2
export DATA_SIZE="32"


for EPOCH in 2
do
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
                --decoder-layers 4 \
                --decoder-embed-dim 128 \
                --decoder-ffn-embed-dim 512 \
                --decoder-attention-heads 2 \
                --decoder-learned-pos \
                --no-scale-embedding \
                --dropout 0.0 \
                --attention-dropout 0.0 \
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
                --lr 0.00025 \
                --end-learning-rate 0.000025 \
                --warmup-updates 2000 \
                --total-num-update 64 \
                --max-update 64 \
                --tokens-per-sample 64 \
                --batch-size 1 \
                --update-freq 1 \
                --log-format json \
                --log-interval 1 \
                --ignore-unused-valid-subsets \
                --validate-interval-updates 100000 \
                --wandb-project Molecular_Generation_with_GDB13 \
                --wandb-run-name OPT_302M_ep_$EPOCH"_aspirin_0.4_sf_"$DATA_SIZE"_"test \
                --save-interval-epochs 1 \
                --keep-last-updates 1 \
                --save-dir ./checkpoints/OPT_302M_ep_$EPOCH"_aspirin_0.4_sf_"$DATA_SIZE \
                --restore-file "" 
done                