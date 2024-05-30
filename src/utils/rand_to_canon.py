import glob
import json
import time
import argparse
from tqdm import tqdm
from chem_utils import get_canonical_form


def make_canonical(input_file, output_file):
    print("Reading from file", input_file)
    
    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            for line_str in tqdm(infile):
                line_obj = json.loads(line_str)
                # Transform to canon
                canon_form = get_canonical_form(line_obj["text"])
                # Write
                new_line = {"text": canon_form}
                json.dump(new_line, outfile)
                outfile.write("\n")
                       

def main():
    parser = argparse.ArgumentParser(description='N/A')

    parser.add_argument('--input_file', type=str, help='Path to the input file')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    args = parser.parse_args()

    print("Processing file: ", args.input_file)
    start_time = time.time()

    make_canonical(args.input_file, args.output_file)

    print("Runtime:", time.time() - start_time)  


if __name__ == "__main__":
    main()
