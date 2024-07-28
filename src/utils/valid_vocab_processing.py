import os
import glob
import json
from tqdm import tqdm
from pathlib import Path
from get_tokenizer import get_tokenizer

def get_full_rand_valid(
    str_type: str,
    str_type_ext: str,
    subset: str
) -> None:
    """
    Create a randomized validation JSONL file from molecule data files.

    Args:
        str_type (str): Type of the molecule data. Options: `selfies` or `smiles`.
        str_type_ext (str): Extension type of the molecule data. Options:`sf` (for `selfies`) or `sm` (for `smiles`).
        subset (str): The name of a subset. Options: `aspirin`, `sas`, `druglike` or `eqdist`.
    """
    subset_dir = Path(f'/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/data/data/randomized/{subset}/')
    rand_output_path = Path(f'/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13/src/data/data/randomized_valids/{str_type}/{subset}/rand_valid_10K_{str_type_ext}.jsonl')

    # Ensure the output directory exists
    rand_output_path.parent.mkdir(parents=True, exist_ok=True)

    files_list = [file for file in glob.glob(os.path.join(subset_dir, '*')) if os.path.isfile(file)]

    with open(rand_output_path, 'w') as out_file:
        for mol_file in tqdm(files_list, desc="Processing molecule files"):
            try:
                with open(mol_file, 'r') as file:
                    mol_id = Path(mol_file).stem
                    mol_data = json.load(file)
                    if mol_id in mol_data:
                        mol_list = mol_data[mol_id].get(str_type, [])
                        for mol_str in mol_list:
                            new_line = {"text": mol_str}
                            json.dump(new_line, out_file)
                            out_file.write("\n")
                    else:
                        print(f"ID {mol_id} not found in file: {mol_file}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {mol_file}")
            except Exception as e:
                print(f"Error processing file {mol_file}: {e}")


def vocab_tokens_to_jsonl(tokenizer_path: str, output_path: str) -> None:
    """
    Convert tokenizer vocabulary tokens to a JSONL file.

    Args:
        tokenizer_path (str): Path to the tokenizer.
        output_path (str): Path to save the JSONL file.
    """
    try:
        tokenizer = get_tokenizer(tokenizer_path=tokenizer_path)
        vocab_dict = tokenizer.get_vocab()
    except Exception as e:
        print(f"Error loading tokenizer or getting vocabulary: {e}")
        return

    try:
        with open(output_path, "w") as file:
            for token in vocab_dict.keys():
                new_line = {"text": token}
                json.dump(new_line, file)
                file.write("\n")
        print(f"Vocabulary tokens successfully written to {output_path}")
    except Exception as e:
        print(f"Error writing tokens to JSONL file: {e}")

tokenizer = '/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13_my/src/data/tokenizers/tokenizer_sf/tokenizer.json'
vocab_tokens_to_jsonl(tokenizer_path=tokenizer, output_path='/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13_my/src/experiments/data/temp_smm.jsonl')