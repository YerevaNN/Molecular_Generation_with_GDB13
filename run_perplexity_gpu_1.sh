export CUDA_VISIBLE_DEVICES="0"

export PROJECT_NAME=Molecular_Generation_with_GDB13 
export DATA_SUBSET=all_sf_valid_100K
export MODEL_NAME="OPT_680M_ep_1_all_sf_256M_2.50E-04"



metaseq-validate --task streaming_language_modeling \
                ../$PROJECT_NAME/data/data_bin_$DATA_SUBSET/ \
                --sample-break-mode "eos_pad_8" \
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
                --batch-size 256 \
                --log-format json \
                --path "./checkpoints/$MODEL_NAME/checkpoint_last.pt" \
                --results-path ./ablations/perplexities/$MODEL_NAME"_"$DATA_SUBSET.csv \
                --return-perplexities