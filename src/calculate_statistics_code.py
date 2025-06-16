import json
import time
import argparse
import pandas as pd
import selfies as sf
from tqdm import tqdm
from rdkit import Chem
import multiprocessing as mp

    
def jsonl_to_list(jsonl_path: str) -> list[str]:
    with open(jsonl_path, 'r') as file:
        print(next(file))
        mol_list = [json.loads(line_str)["text"] for line_str in tqdm(file)]

        return mol_list


def csv_to_list(csv_path: str) -> list[str]:
    with open(csv_path, 'r') as file:
        mol_list = []

        for line in tqdm(file):
            line = line.strip()

            if "," in line:
                mol_list.append(line.split(",")[0])
            else:
                mol_list.append(line)    

    return mol_list


def convert_to_canonical_smiles(mol_str: str) -> str:
    try:
        mol = Chem.MolFromSmiles(mol_str)
        if mol is None:
            raise ValueError
        else:
            canon_smiles = Chem.MolToSmiles(mol)
    except ValueError:
        print(f'Invalid SMILES {mol_str}')
        return None
    else:
        return canon_smiles
    

def convert_to_canonical_selfies(mol_str: str) -> str:
    """
    Converts a molucular 1D representations (selfies) into canonical selfies representation.

    Args:
        mol_str (string): A molucular 1D representations (selfies) to be converted.

    Returns:
        list: A canonical molucular 1D representations (selfies) corresponding to the input if it is possible, otherwise None.

    Note:
        Requires the 'Chem' library for smiles processing and the 'sf' module for selfies encoding/decoding.
    """
    try:
        smiles = sf.decoder(mol_str)
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError
        else:
            canon_smiles = Chem.MolToSmiles(mol)
            canon_selfies = sf.encoder(canon_smiles)

    except ValueError:
        print(f'Invalid SMILES {mol_str}')

    except sf.exceptions.EncoderError as e:
        print("Error encoding SMILES string:", e, mol_str)

    except sf.exceptions.DecoderError as e:
        print("Error decoding SELFIES string:", e, mol_str)
    
    else:
        # with open("/auto/home/knarik/Molecular_Generation_with_GDB13/src/ablations/generations/generations/sf/generations_10M/unigram/rand_equal_dist_valid_8M.csv", "a") as f_w:
        #     # print("write", mol_str)
        #     f_w.write(mol_str + "\n")    
        return canon_selfies
    

def calculate_statistics(num_processes: int, chunk_size: int, gen_length: int, mol_repr: str, subset_list: list, generation_list: list) -> tuple[int,int]:
    """
    Calculates the number of duplicated and unique generated molecules from subset.

    Args:
        num_processes (int): The number of worker processes for parallelization.
        chunk_size (int): The number of molecules to be converted to canonical form parallelly.
        mol_repr (str): The type of molecular representation (selfies/smiles).
        subset_list (list): A list containing the molecules of the subset.
        generation_list (list): A list containing the generated molecules.

    Returns: 
        tuple[int,int]: A tuple consisting of the number of duplicated and unique generated molecules from subset.
    """
    ctx = mp.get_context('spawn')

    # Convert generations to canonical forms
    convert_to_canonical = convert_to_canonical_selfies if mol_repr == 'sf' else convert_to_canonical_smiles

    print('Start to convert generation into its canonical.')
    time1 = time.time()
    canon_gen_list = []

    with ctx.Pool(num_processes) as p:
        for i in range(0, gen_length, chunk_size):
            canon_gen_list += p.map(convert_to_canonical, generation_list[0 + i : chunk_size + i])

    print('Time:', time.time() - time1)
    print('canon_gen_length', len(canon_gen_list))
    print('generation length', len(generation_list))

    print('Start to calculate the number of unique true positives.')
    time1 = time.time()

    intersect_subset_gen = set(subset_list) & set(canon_gen_list)
    num_unique_tp = len(intersect_subset_gen)

    print('Time:', time.time() - time1)

    print('Start to calculate the number of duplicated true posities.')
    time1 = time.time()
    gen_dict = {}

    for mol in canon_gen_list:
        gen_dict[mol] = gen_dict.get(mol, 0) + 1

    num_tp = sum(gen_dict[mol] for mol in intersect_subset_gen)
    print('Time:', time.time() - time1)

    return num_tp, num_unique_tp


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate statistics on generation")
    parser.add_argument(
        "--mol_repr",
        type=str,
        default=None,
        help="The type of molecular representation (sf/sm).",
    )
    parser.add_argument(
        "--subset_path",
        type=str,
        default=None,
        help="The path to the file containing the molecules of the subset.",
    )
    parser.add_argument(
        "--generation_path",
        type=str,
        default=None,
        help="A path to the file containing the generated molecules.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The path to the file where the statistics will be saved.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1,
        help="The number of molecules to be converted to canonical form parallelly."
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="The number of worker processes for parallelization."
    )
    parser.add_argument(
        "--gen_length",
        type=int,
        default=1_000_000,
        help="The number of generated strings."
    )
    args = parser.parse_args()

    return args
   
   
if __name__=='__main__':
    args = parse_args()

    time1 = time.time()
    print("Start to collect subset in a list")
    subset_list = jsonl_to_list(args.subset_path)
    print(time.time() - time1)
    print(len(subset_list))
    print(subset_list[:2])
    print("===============")
    time1 = time.time()
    print("Start to collect generation in a list")
    generation_list = csv_to_list(args.generation_path)
    print(time.time() - time1)
    print(len(generation_list))
    print(generation_list[:2])

    exit()

    time1 = time.time()
    num_tp, num_unique_tp = calculate_statistics(
        num_processes=args.num_processes,
        chunk_size=args.chunk_size,
        gen_length=args.gen_length,
        mol_repr=args.mol_repr,
        subset_list=subset_list, 
        generation_list=generation_list
        )
    
    print(f'Write into a file {args.output_path}')
    stat_df = pd.DataFrame({'TP': [num_tp], "Unique TP": [num_unique_tp]})
    stat_df.to_excel(args.output_path, index=False)

    print(stat_df)
    print(time.time() - time1)