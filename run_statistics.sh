export TRAIN_FOLDER="aspirin_0.4"
export COLUMN_NAME="aspirin_similarity" # aspirin_similarity / sascore / drug_sim / poison_sim
export CONDITION=">=0.4" # >=0.4, <=3
export GEN_LEN=100000

export FILE_PATH=./ablations/generations/OPT_1.2B_ep_1_$TRAIN_FOLDER"_sf_1000K_gen_1000000.csv"
export OUTPUT_NAME=OPT_1.2B_ep_1_$TRAIN_FOLDER"_sf_1000K_gen_"$GEN_LEN
export OUTPUT_FILE=./ablations/statistics/Sampling_results_$OUTPUT_NAME".xlsx"
export SCORES_OUTPUT_FILE=./ablations/statistics/Scores_$OUTPUT_NAME.csv
export DB_FILE_PATH=/mnt/xtb/knarik/Smiles.db


python get_stats_subsets.py \
    --file_path $FILE_PATH \
    --train_folder $TRAIN_FOLDER \
    --column_name $COLUMN_NAME \
    --condition $CONDITION \
    --gen_len $GEN_LEN \
    --output_file $OUTPUT_FILE \
    --db_path $DB_FILE_PATH \
    --scores_output_file $SCORES_OUTPUT_FILE \
 