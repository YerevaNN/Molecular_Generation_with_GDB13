import os 
import json
import random
import argparse
import parmap
import time
import selfies as sf
from rdkit import Chem
from tqdm import tqdm
import concurrent


def create_dir(file_path):
    dirname = os.path.dirname(file_path)

    if not os.path.exists(dirname):
        os.makedirs(dirname)


def write_jsonl(line_str, output_file):
    new_line = {"text": line_str}
    
    json.dump(new_line, output_file)
    output_file.write("\n")


def randomize_smiles(smiles_str, random_type="unrestricted"):
    mol = None

    # try:    
        # Transform to molecule
    mol = Chem.MolFromSmiles(smiles_str)
    # except:
    #     print("Can't parse smiles")    

    if not mol:
        return None

    if random_type == "unrestricted":
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    
    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumHeavyAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
    
    raise ValueError("Type '{}' is not valid".format(random_type))


def randomize_selfies(selfies_str: str, random_type="restricted"):
    rand_selfies_str = ""

    # try:
    # Transform to smiles
    smiles_str = sf.decoder(selfies_str)

    # rand_smiles_str = smiles_str
    # counter = 0
    
    # while smiles_str == rand_smiles_str:
    #     # Randomize
    #     counter += 1
    rand_smiles_str = randomize_smiles(smiles_str) 

    # if counter > 1:
    #     print("The same smiles", counter)    

    # Transform to selfies (randomized)
    rand_selfies_str = sf.encoder(rand_smiles_str)
    # except Exception as e:
    #     print(e)

    return rand_selfies_str 


def read_lines_from_file(rand_number, input_file_name, output_file_name):
    with open(input_file_name, 'r') as input_file:
        with open(output_file_name, "w") as output_file:
            count = 0

            for line_str in tqdm(input_file):
                # Get the string line
                line_obj = json.loads(line_str)
                selfies_str = line_obj["text"]

                # Write down an initial selfies
                write_jsonl(selfies_str, output_file)

                # Container to count all unique rand_number randomized versions     
                randomized_set = set()
                randomized_set.add(selfies_str)      
                check_iter = 0

                while len(randomized_set) < rand_number:
                    check_iter += 1

                    proceed_selfies_str = randomize_selfies(selfies_str)

                    if proceed_selfies_str not in randomized_set:
                        # Write down only unique ones
                        write_jsonl(proceed_selfies_str, output_file)
                        randomized_set.add(proceed_selfies_str)
                        
                    if check_iter > 1000:
                        print(f"Sorry, too many iterations for finding all {rand_number} randomized versions for {selfies_str}, found {len(randomized_set)}.")
                        count += 1
                        number_to_copy = rand_number - len(randomized_set)

                        if number_to_copy > len(randomized_set):
                            multiplied_randomized_list = number_to_copy * list(randomized_set)
                            copied_elements = multiplied_randomized_list[:number_to_copy]
                        else:    
                            copied_elements = list(randomized_set)[:number_to_copy]

                        while copied_elements:
                            # Just copy first elements if there is no so many versions of randomized
                            write_jsonl(copied_elements.pop(), output_file)

                        break

            print(f"Overall count that don't have enough randomized is {count}")    
                        

def main():
    parser = argparse.ArgumentParser(description='N/A')

    parser.add_argument('--rand_number', type=int, help='The count of the randomized strings.')
    parser.add_argument('--input_file', type=str, help='Path to the input file')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    args = parser.parse_args()
    
    rand_number = args.rand_number
    input_file_path = args.input_file
    output_file_path = args.output_file
 
    print("Processing file: ", input_file_path)
    start_time=time.time()

    # Create directory if doesn't exist
    create_dir(output_file_path)

    # Create randomized versions
    read_lines_from_file(rand_number, input_file_path, output_file_path)

    print("Overall takes", time.time()-start_time)  


if __name__ == "__main__":
    main()