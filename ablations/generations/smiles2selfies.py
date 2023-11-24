import os 
import json
import argparse
import selfies as sf
from rdkit import Chem
from tqdm import tqdm


def smiles2selfies(input_file_path, output_file_path):
    with open(input_file_path, "r") as file_1:
        data = file_1.read().splitlines()
        with open(output_file_path, "w") as file_2:
            for line_str in tqdm(data):
                # line_obj = json.loads(line_str)
                # Transform to selfies
                # selfies_str = sf.encoder(line_obj["text"])
                selfies_str = sf.encoder(line_str)
                # Write
                new_line = {"text": selfies_str}
                json.dump(new_line, file_2)
                file_2.write("\n")


def main():
    parser = argparse.ArgumentParser(description='N/A')

    parser.add_argument('--input_file', type=str, help='Path to the input file')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    args = parser.parse_args()
    input_file_path = args.input_file
    output_file_path = args.output_file

    print("Processing file: ", input_file_path)
    smiles2selfies(input_file_path, output_file_path)


if __name__ == "__main__":
    main()