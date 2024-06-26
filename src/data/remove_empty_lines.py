from tqdm import tqdm
import glob
import json

def remove_empty(input_file, output_file):
    print("Reading from file", input_file)
    
    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            for i, line in tqdm(enumerate(infile)):
                if line.strip() == "":
                    print("Empty line at", i)
                else:    
                    outfile.write(line)
                       


if __name__ == "__main__":
    input_file = "../src/data/data/train_all_rand_sm_848M.jsonl"
    output_file = "../src/data/data/train_all_rand_sm_848M_no_empty.jsonl"

    remove_empty(input_file, output_file)

    # use "tail data_bin_pretrain_1B.jsonl -n 5" to check last rows
    # use "truncate -s -1 data_bin_pretrain_1B.jsonl" to remove last row if it is an empty line
    # use "shuf input_file -o output_file" to shuffle the data