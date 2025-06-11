import json
import random
import selfies as sf
from tqdm import tqdm
from rdkit import Chem
from typing import Set


def sf_to_sm(selfies: str, canon: bool = False) -> str:
    """
    Convert a SELFIES string to a SMILES string.

    Args:
        selfies (str): SELFIES string.
        canon (bool): Whether to return canonical SMILES. Default is False.

    Returns:
        str: SMILES string.
    """
    try:
        smiles = sf.decoder(selfies)
        if canon:
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Error converting SELFIES to SMILES for {selfies}: {e}")
    return smiles


def sm_to_sf(smiles: str, canon: bool = False) -> str:
    """
    Convert a SMILES string to a SELFIES string.

    Args:
        smiles (str): SMILES string.
        canon (bool): Whether to use canonical SMILES. Default is False.

    Returns:
        str: SELFIES string.
    """
    try:
        if canon:
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol)
        selfies = sf.encoder(smiles)
    except Exception as e:
        print(f"Error converting SMILES to SELFIES for {smiles}: {e}")
    return selfies


def sf_to_sm_jsonl(sf_path: str, sm_path: str, canon: bool = False) -> None:
    """
    Convert a JSONL file of SELFIES strings to a JSONL file of SMILES strings.

    Args:
        sf_path (str): Path to the input JSONL file with SELFIES strings.
        sm_path (str): Path to the output JSONL file with SMILES strings.
        canon (bool): Whether to return canonical SMILES. Default is False.
    """
    try:
        with open(sm_path, 'w') as sm_file:
            with open(sf_path, 'r') as sf_file:
                for line_str in tqdm(sf_file, desc="Converting SELFIES to SMILES"):
                        selfies = json.loads(line_str)["text"]
                        smiles = smiles = sf_to_sm(selfies=selfies, canon=canon)
                        new_line = {"text": smiles}
                        json.dump(new_line, sm_file)
                        sm_file.write('\n')
    except Exception as e:
        print(f"Error writing SMILES JSONL file: {e}")


def sm_to_sf_jsonl(sm_path: str, sf_path: str, canon: bool = False) -> None:
    """
    Convert a JSONL file of SMILES strings to a JSONL file of SELFIES strings.

    Args:
        sm_path (str): Path to the input JSONL file with SMILES strings.
        sf_path (str): Path to the output JSONL file with SELFIES strings.
        canon (bool): Whether to use canonical SMILES. Default is False.
    """
    try:
        with open(sf_path, 'w') as sf_file:
            with open(sm_path, 'r') as sm_file:
                for line_str in tqdm(sm_file, desc="Converting SMILES to SELFIES"):
                        smiles = json.loads(line_str)["text"]
                        selfies = sm_to_sf(smiles=smiles, canon=canon)
                        new_line = {"text": selfies}
                        json.dump(new_line, sf_file)
                        sf_file.write('\n')
    except Exception as e:
        print(f"Error writing SELFIES JSONL file: {e}")

    
def rand_to_canon_str(sequence: str, str_type: str) -> str:
    """
    Convert a random sequence to its canonical form.

    Args:
        sequence (str): The input sequence, either SELFIES or SMILES.
        str_type (str): The type of the sequence ('selfies' or 'smiles').

    Returns:
        str: The canonical form of the sequence.

    Raises:
        ValueError: If the provided str_type is unsupported.
    """
    if str_type == 'sf':
        smiles = sf.decoder(sequence)
    elif str_type == 'sm':
        smiles = sequence
    else:
        raise ValueError(f"Unsupported str_type: {str_type}")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    canon_smiles = Chem.MolToSmiles(mol)
    
    if str_type == 'sf':
        return sf.encoder(canon_smiles)
    else:
        return canon_smiles

def rand_to_canon_jsonl(rand_path: str, canon_path: str, str_type: str) -> None:
    """
    Convert sequences in a JSONL file from random to canonical form.

    Args:
        rand_valid_path (str): Path to the input JSONL file with random sequences.
        canon_valid_path (str): Path to the output JSONL file with canonical sequences.
        str_type (str): The type of the sequences ('selfies' or 'smiles').

    Raises:
        ValueError: If an invalid sequence or unsupported str_type is encountered.
    """
    try:
        with open(canon_path, 'w') as canon_file:
            with open(rand_path, 'r') as rand_file:
                for line_str in tqdm(rand_file, desc="Converting to canonical sequences"):
                    sequence = json.loads(line_str)["text"]
                    canon_str = rand_to_canon_str(sequence=sequence, str_type=str_type)
                    new_line = {"text": canon_str}
                    json.dump(new_line, canon_file)
                    canon_file.write('\n')
    except Exception as e:
        print(f"Error writing canonical JSONL file: {e}")


def is_canon(str_repr: str, str_type: str = 'smiles') -> bool:
    """
    Check if a given string representation is in canonical form.

    Args:
        str_repr (str): The input string representation (SMILES or SELFIES).
        str_type (str): The type of the representation ('smiles' or 'selfies').

    Returns:
        bool: True if the representation is canonical, False otherwise.
    """
    if str_type == 'sf':
        smiles = sf.decoder(str_repr)
    elif str_type == 'sm':
        smiles = str_repr
    else:
        raise ValueError(f"Unsupported str_type: {str_type}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    canon_smiles = Chem.MolToSmiles(mol)

    if str_type == 'sf':
        canon_selfies = sf.encoder(smiles=canon_smiles)
        return canon_selfies == str_repr

    return canon_smiles == smiles


def randomize_string(str_repr: str, str_type: str = 'smiles') -> str:
    """
    Generate a random SMILES given a SMILES or SELFIES representation of a molecule.

    Args:
        str_repr (str): The input string representation (SMILES or SELFIES).
        str_type (str): The type of the representation ('smiles' or 'selfies').

    Returns:
        str: A random SMILES or SELFIES string of the same molecule, or None if the molecule is invalid.
    """
    if str_type == 'sf':
        smiles = sf.decoder(str_repr)
    elif str_type == 'sm':
        smiles = str_repr
    else:
        raise ValueError(f"Unsupported str_type: {str_type}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    new_atom_order = list(range(mol.GetNumAtoms()))
    random.shuffle(new_atom_order)
    random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
    random_smiles = Chem.MolToSmiles(random_mol, canonical=True, doRandom=True, isomericSmiles=False)

    if str_type == 'sf':
        return sf.encoder(random_smiles)
    
    return random_smiles


def get_rand_set(str_repr: str, str_type: str = 'smiles', iter_len: int = 100000) -> Set[str]:
    """
    Generate a set of random string representations of a molecule.

    Args:
        str_repr (str): The input string representation (SMILES or SELFIES).
        str_type (str): The type of the representation ('smiles' or 'selfies').
        iter_len (int): The number of random representations to generate.

    Returns:
        Set[str]: A set of random string representations.
    """
    rand_set = set()
    for _ in range(iter_len):
        rand_set.add(randomize_string(str_repr=str_repr, str_type=str_type))
    return rand_set