import os 
import json
import argparse
import time
import selfies as sf
from rdkit import Chem
from tqdm import tqdm

def create_dir(file_path):
    dirname = os.path.dirname(file_path)

    if not os.path.exists(dirname):
        os.makedirs(dirname)


def write_jsonl(line_str, output_file):
    new_line = {"text": line_str}
    
    json.dump(new_line, output_file)
    output_file.write("\n")


def read_lines_from_file(input_file_name, output_file_name):
    with open(input_file_name, 'r') as input_file:
        with open(output_file_name, "w") as output_file:

            for line_str in tqdm(input_file):
                # Get the string line
                line_obj = json.loads(line_str)
                rand_selfies_str = line_obj["text"]

                # Transform to smiles
                rand_smiles_str = sf.decoder(rand_selfies_str)

                try:    
                    # Transform to molecule
                    mol = Chem.MolFromSmiles(rand_smiles_str)
                except:
                    print("Can't parse smiles")  
                    continue

                canon_smiles_str = Chem.MolToSmiles(mol, canonical=True)


                # Transform to selfies (canon)
                canon_selfies_str = sf.encoder(canon_smiles_str)


                # Write down
                write_jsonl(canon_selfies_str, output_file)
                        

def main():
    parser = argparse.ArgumentParser(description='N/A')

    parser.add_argument('--input_file', type=str, help='Path to the input file')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    args = parser.parse_args()
    
    input_file_path = args.input_file
    output_file_path = args.output_file
 
    print("Processing file: ", input_file_path)
    start_time=time.time()

    # Create directory if doesn't exist
    create_dir(output_file_path)

    # Create randomized versions
    read_lines_from_file(input_file_path, output_file_path)

    print("Overall takes", time.time()-start_time)  


if __name__ == "__main__":
    main()