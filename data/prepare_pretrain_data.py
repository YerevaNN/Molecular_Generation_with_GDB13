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


def smiles2selfies(path):
    with open(path, "r") as file_1:
        path_2_name = os.path.basename(path)
        with open(path_2_name, "w") as file_2:
            for line_str in tqdm(file_1):
                line_obj = json.loads(line_str)
                # Transform to selfies
                selfies_str = sf.encoder(line_obj["text"])
                # Write
                new_line = {"text": selfies_str}
                json.dump(new_line, file_2)
                file_2.write("\n")


def selfies2smiles(input_file_path, output_file_path):
    with open(input_file_path, "r") as file_1:
        with open(output_file_path, "w") as file_2:
            for line_str in tqdm(file_1):
                line_obj = json.loads(line_str)
                # Transform to smiles
                try:    
                    smiles_str = sf.decoder(line_obj["text"])
                    canon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_str))
                    # Write
                    new_line = {"text": canon_smiles}
                    json.dump(new_line, file_2)
                    file_2.write("\n")
                except:
                    print("Can't convert.")    


def randomize_smiles(smiles_str, random_type="restricted"):
    mol = None

    try:    
        # Transform to molecule
        mol = Chem.MolFromSmiles(smiles_str)
    except:
        print("Can't parse smiles")    

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


def selfies2randomized(input_file_path, output_file_path, rand=False):
    with open(input_file_path, "r") as file_1:
        data = file_1.read().splitlines()

    lines = []

    with open(output_file_path, "w") as file_2:
        for line in tqdm(data):
            lines.append(process_fn(line, rand))

        # # lines = parmap.map(process_fn, data_half, pm_processes=2)
        # print("Processed lines count", len(lines))

        print("Writing in a file", output_file_path)
        for line in tqdm(lines):
            json.dump(line, file_2)
            file_2.write("\n")


def randomize_selfies(selfies_str: str, random_type="restricted"):
    rand_selfies_str = ""

    try:
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
    except Exception as e:
        print(e)

    return rand_selfies_str 


def process_fn(line_str, prefix, suffix, rand):
    try:
        line_obj = json.loads(line_str)

        selfies_str = line_obj["text"]
        if rand:
            proceed_selfies_str = randomize_selfies(selfies_str)
        else:
            proceed_selfies_str = selfies_str    

        if proceed_selfies_str:
            # Write
            new_line = {"text": prefix + proceed_selfies_str + suffix}

            return new_line
    except Exception as e:
        print(e)
        print("Line string", line_str)

    return None             


def read_lines_from_file(start_line, num_lines, input_file_name, output_file_name, prefix, suffix, rand):
    start_time = time.time()

    with open(input_file_name, 'r') as file:
        # Skip to the start_line
        for _ in range(start_line - 1):
            next(file)

        print(f"Command Line Arguments: Start Line: {start_line}, Number of Lines: {num_lines}, File Name: {input_file_name}. Time taken to skip to start line: {time.time() - start_time:.2f} seconds")
        
        with open(output_file_name, "w") as file_2:
            for _ in range(num_lines):
                line = next(file, None)

                if line is None: 
                    break
    
                proceed_line = process_fn(line, prefix, suffix, rand)  
    
                if proceed_line:
                    json.dump(proceed_line, file_2)
                    file_2.write("\n")
     

def main():
    parser = argparse.ArgumentParser(description='N/A')

    parser.add_argument('--start', type=int, help='Start line in the file.')
    parser.add_argument('--increment', type=int, help='Chunk size.')
    parser.add_argument('--input_file', type=str, help='Path to the input file')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    parser.add_argument('--prefix', default="", type=str, help='Can be any, but we use [Canon] / [Rand]')
    parser.add_argument('--suffix', default="", type=str, help='Can be any, but we use [Canon] / [Rand]')
    parser.add_argument('--rand', default=False, action='store_true', help='Randomized or not.')
    args = parser.parse_args()
    
    start = args.start
    rand = args.rand
    increment = args.increment
    input_file_path = args.input_file
    output_file_path = args.output_file + str(start) + ".jsonl"
    suffix = args.suffix
    prefix = args.prefix

    print("Processing file: ", input_file_path)
    start_time=time.time()
    read_lines_from_file(start, increment, input_file_path, output_file_path, prefix, suffix, rand)

    print(time.time()-start_time)  


if __name__ == "__main__":
    main()