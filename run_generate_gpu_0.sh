export CUDA_VISIBLE_DEVICES="0"


# generation parameters
export MAX_SEQ_LEN=64
export BATCH_SIZE=256
export GEN_LEN=1000000


# sampling parameters
export MAX_BEAM=1
export SAMPLING_TOPP=1
export TEMPERATURE=1
export LOGPROBS=0
export DESCRIPTION=gen_$GEN_LEN   # temp/topp


# model parameters
export DATA_SUBSET=all
export DATA_SIZE="64M"
export MOL_REPR='sf'  # sf/sm
export MODEL_SIZE="800K"
export EPOCH=1


# paths
export PROJECT=../Molecular_Generation_with_GDB13
export DATA=/tmp  # required, it doesn't matter what to write
export FOLDER=OPT_$MODEL_SIZE"_ep_"$EPOCH"_"$DATA_SUBSET"_"$MOL_REPR"_"$DATA_SIZE
export OUTPUT_FILE_PATH=$PROJECT/ablations/generations/$FOLDER"_"$DESCRIPTION".csv"



# tokenizer config
# if [ "$MOL_REPR" == "sf" ]; then
export TOKENIZER="--hf-tokenizer  $PROJECT/data/tokenizers/tokenizer_sf/tokenizer.json"
# else
#     export TOKENIZER="--vocab-filename $PROJECT/data/tokenizers/tokenizer_sm/vocab.txt"
# fi


for model_size in "85M"
do  
    export FOLDER=OPT_$model_size"_ep_"$EPOCH"_"$DATA_SUBSET"_"$MOL_REPR"_"$DATA_SIZE
    export OUTPUT_FILE_PATH=$PROJECT/ablations/generations/$FOLDER"_"$DESCRIPTION".csv"

    python  ../metaseq/metaseq/cli/interactive_cli_simple.py \
        --task language_modeling \
        $TOKENIZER \
        --path $PROJECT/checkpoints/$FOLDER/checkpoint_last.pt \
        --beam $MAX_BEAM \
        --sampling-topp $SAMPLING_TOPP \
        --temperature $TEMPERATURE \
        --logprobs $LOGPROBS\
        --sampling \
        --bpe hf_byte_bpe \
        --generation-len $GEN_LEN \
        --output-file-path $OUTPUT_FILE_PATH \
        --mol-repr $MOL_REPR \
        --batch-size $BATCH_SIZE \
        --buffer-size $BATCH_SIZE * $MAX_SEQ_LEN \
        --max-tokens $BATCH_SIZE * $MAX_SEQ_LEN 
done   