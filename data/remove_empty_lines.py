from tqdm import tqdm
import glob
import json

def remove_empty(input_file, output_file):
    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            for i, line in tqdm(enumerate(infile)):
                if line.strip()=="":
                    print(i)
                else:    
                    outfile.write(line)
                       


if __name__ == "__main__":
    input_file = "train_half_rand_sf_848M.jsonl"
    output_file = "train_half_rand_sf_848M_no_empty.jsonl"

    remove_empty(input_file, output_file)

    # use "tail data_bin_pretrain_1B.jsonl -n 5" to check last rows
    # use "truncate -s -1 data_bin_pretrain_1B.jsonl" to remove last row if it is an empty line
    # use "shuf input_file -o output_file" to shuffle the data