export CUDA_VISIBLE_DEVICES="0"
export CUDA_LAUNCH_BLOCKING=1

export PROJECT_NAME=Molecular_Generation_with_GDB13 

metaseq-validate --task streaming_language_modeling \
                ../$PROJECT_NAME/data/data_bin_aspirin_train_sf/ \
                --sample-break-mode "passthrough" \
                --hf-tokenizer ../$PROJECT_NAME/data/tokenizers/tokenizer_sf/tokenizer.json \
                --train-subset train \
                --valid-subset valid \
                --combine-valid-subsets \
                --no-reshard-after-forward \
                --use-sharded-state \
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
                --tokens-per-sample 64 \
                --batch-size 1 \
                --log-format json \
                --path "./ablations/perplexities/aspirin_0.4_sf/checkpoint_180000.pt" \
                --results-path "./ablations/perplexities/aspirin_0.4_sf/OPT_iter_180000_train_pp.csv" \
                --return-perplexities