import json
import time
import argparse
import pandas as pd
import selfies as sf
from tqdm import tqdm
from rdkit import Chem
import multiprocessing as mp
    
def jsonl_to_list(jsonl_path: str) -> list[str]:
    """
    Converts the file from input jsonl file to a list.

    Args:
        jsonl_path (jsonl): A jsonl file containing molucular 1D representations (selfies/smiles).

    Returns:
        list: A list of molucular 1D representations corresponding to the input.
    """
    with open(jsonl_path, 'r') as file:
        mol_list = [json.loads(line_str)["text"] for line_str in tqdm(file)]
        return mol_list


def csv_to_list(csv_path: str) -> list[str]:
    """
    Converts the file from input csv file to a list.

    Args:
        csv_path (csv): A csv file containing generated molucular 1D representations (selfies/smiles). 

    Returns:
        list: A list of molucular 1D representations (selfies/smiles) corresponding to the input.
    """
    with open(csv_path, 'r') as file:
        mol_list = [mol_str.strip() for mol_str in tqdm(file) if mol_str.strip()]
    return mol_list


def convert_to_canonical_smiles(mol_str: str) -> str:
    """
    Converts A molucular 1D representations (smiles) into canonical canonical smiles representation.

    Args:
        mol_str (string): A molucular 1D representations (smiles) to be converted.

    Returns:
        list: A canonical molucular 1D representations (smiles) corresponding to the input if it is possible, otherwise None.

    Note:
        Requires the 'Chem' library for smiles processing.
    """
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
        print("Error encoding SMILES string:", e, canon_smiles)

    except sf.exceptions.DecoderError as e:
        print("Error decoding SELFIES string:", e, mol_str)
    
    else:
        return canon_selfies
    

def calculate_statistics(num_processes: int, chunk_size: int, gen_length: int, str_type: str, subset_list: list, generation_list: list) -> tuple[int,int]:
    """
    Calculates the number of duplicated and unique generated molecules from subset.

    Args:
        num_processes (int): The number of worker processes for parallelization.
        chunk_size (int): The number of molecules to be converted to canonical form parallelly.
        str_type (str): The type of molecular representation (selfies/smiles).
        subset_list (list): A list containing the molecules of the subset.
        generation_list (list): A list containing the generated molecules.

    Returns: 
        tuple[int,int]: A tuple consisting of the number of duplicated and unique generated molecules from subset.
    """
    ctx = mp.get_context('spawn')

    # Convert generations to canonical forms
    if str_type == 'selfies':
        convert_to_canonical_representation = convert_to_canonical_selfies
    elif str_type == 'smiles':
        convert_to_canonical_representation = convert_to_canonical_smiles
    else:
        print('Representation must be either smiles or selfies')

    print('Start to convert generation into its canonical.')
    time1 = time.time()
    canon_gen_list = []
    with ctx.Pool(num_processes) as p:
        for i in range(0, gen_length, chunk_size):
            canon_gen_list += p.map(convert_to_canonical_representation, generation_list[0+i:chunk_size+i])
    print('Done. Time:', time.time()-time1)
    print('canon_gen_length', len(canon_gen_list))
    print('generation length', len(generation_list))

    # Get the number of unique molecules generated from subset
    print('Start to calculate the number of unique true positives.')
    time1 = time.time()
    intersect_subset_gen = set(subset_list) & set(canon_gen_list)
    num_unique_tp = len(intersect_subset_gen)
    print('Done. Time:', time.time() - time1)

    # Get the number of all molecules generated from subset
    print('Start to calculate the number of duplicated true posities.')
    time1 = time.time()
    gen_dict = {}
    for mol in canon_gen_list:
        if mol in gen_dict:
            gen_dict[mol] += 1
        else:
            gen_dict[mol] = 1

    num_tp = sum(gen_dict[mol] for mol in intersect_subset_gen)
    print('Done. Time:', time.time() - time1)

    return num_tp, num_unique_tp


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate statistics on generation")
    parser.add_argument(
        "--str_type",
        type=str,
        default=None,
        help="The type of molecular representation (selfies/smiles).",
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

    # Representation of molecules 'SELFIES' or 'SMILES'
    STR_TYPE = args.str_type 

    # Paths
    SUBSET_PATH = args.subset_path
    GEN_PATH = args.generation_path
    OUTPUT_PATH = args.output_path
    
    GEN_LENGTH = args.gen_length
    CHUNK_SIZE = args.chunk_size
    NUM_PROCESSES = args.num_processes

    # Get the list of strings from subset's file
    time1 = time.time()
    print("Start to collect subset in a list")
    SUBSET_LIST = jsonl_to_list(SUBSET_PATH)
    print('Collecting subset done. Time:', time.time() - time1)

    # Get the list of strings from generations's files
    time1 = time.time()
    print("Start to collect generation in a list")
    GENERATION_LIST = csv_to_list(GEN_PATH)
    print('Collecting generation done. Time:', time.time() - time1)

    time1 = time.time()
    print('Start to calculate statistics')
    num_tp, num_unique_tp = calculate_statistics(num_processes=NUM_PROCESSES, chunk_size=CHUNK_SIZE, gen_length=GEN_LENGTH, str_type=STR_TYPE, subset_list=SUBSET_LIST, generation_list=GENERATION_LIST)
    stat_df = pd.DataFrame({'TP': [num_tp], "Unique TP": [num_unique_tp]})
    stat_df.to_excel(OUTPUT_PATH, index=False)
    print(stat_df)
    print('Calculating statistics done. Time:', time.time() - time1)