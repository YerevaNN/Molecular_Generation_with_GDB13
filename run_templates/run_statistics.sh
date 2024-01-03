export TRAIN_FOLDER="all" # aspirin_0.4, sas_3, druglike_0.4, equal_dist 
export COLUMN_NAME="" # GDB13.aspirin_similarity / GDB13.sascore / GDB13.drug_sim,GDB13.poison_sim
export CONDITION="" # GDB13.aspirin_similarity>=0.4, GDB13.sascore<=3, GDB13.drug_sim>0.4#AND#GDB13.poison_sim<=0.4, GDB13.drug_sim>=0.20#AND#GDB13.drug_sim<0.2165#AND#GDB13.poison_sim>=0.20#AND#GDB13.poison_sim<0.2165
export GEN_LEN=10000

export FILE_PATH=./ablations/generations/OPT_1.2B_ep_1_$TRAIN_FOLDER"_sf_848M_gen_10000.csv"
export OUTPUT_NAME=OPT_1.2B_ep_1_$TRAIN_FOLDER"_sf_848M_gen_"$GEN_LEN
export OUTPUT_FILE=./ablations/statistics/Sampling_results_$OUTPUT_NAME".xlsx"
export SCORES_OUTPUT_FILE=./ablations/statistics/Scores_$OUTPUT_NAME.csv
export DB_FILE_PATH=../../../GDB13.db


python get_stats_subsets.py \
    --file_path $FILE_PATH \
    --train_folder $TRAIN_FOLDER \
    --column_name "$COLUMN_NAME" \
    --condition $CONDITION \
    --gen_len $GEN_LEN \
    --output_file $OUTPUT_FILE \
    --db_path $DB_FILE_PATH \
    --scores_output_file $SCORES_OUTPUT_FILE \
