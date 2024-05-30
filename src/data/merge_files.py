from tqdm import tqdm
import glob

def merge_files(input_files, output_file, chunk_size=4096):
    with open(output_file, 'wb') as outfile:
        for input_file in tqdm(input_files):
            with open(input_file, 'rb') as infile:
                while True:
                    chunk = infile.read(chunk_size)
                    if not chunk:
                        break
                    outfile.write(chunk)
                outfile.write(b"\n")

if __name__ == "__main__":
    input_files = glob.glob("./data/canon_*.jsonl")
    output_file = "./data/train_all_canon_sm_848M.jsonl"

    merge_files(input_files, output_file)

    # use "tail data_bin_pretrain_1B.jsonl -n 5" to check last rows
    # use "truncate -s -1 data_bin_pretrain_1B.jsonl" to remove last row if it is an empty line
    # use "shuf input_file -o output_file" to shuffle the data