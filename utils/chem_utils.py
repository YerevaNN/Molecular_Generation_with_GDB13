import os
import sys
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import selfies as sf
from tqdm import tqdm

# rdkit warnings
import rdkit.RDLogger as rkl
import rdkit.rdBase as rkrb
rkrb.DisableLog('rdApp.error')
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"


def check_validness(smiles: str) -> bool:
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except:
        return False


def check_canonical(smiles: str) -> bool:
    try:
        return get_canonical_form(smiles) == smiles
    except:
        return False
    

def get_canonical_form(smiles: str):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    except:
        print(f"Error while getting the canonical form for {smiles}.")
        return ""   

def get_selfies_form(smiles: str):
    try:
        return sf.encoder(smiles)
    except:
        return "" 
    
def get_smiles_from_selfies(selfies: str):
    canon_smiles = None
    smiles = sf.decoder(selfies)

    try:
        canon_smiles = get_canonical_form(smiles)
    except:
        print("Can't make canonical", selfies)
        return ""

    return canon_smiles    


def get_maccs_fp(smiles: str):
    try: 
        mol = Chem.MolFromSmiles(smiles)

        return MACCSkeys.GenMACCSKeys(mol)
    except:
        print("Can't calculate Maccs fingerprint")
        return ""


def calculate_similarity(smiles: str, property_molecule_smiles: str):
    # Get fingerprints
    smiles_fsp = get_maccs_fp(smiles)
    prop_mol_fsp = get_maccs_fp(property_molecule_smiles)

    if smiles_fsp and prop_mol_fsp:
        # Calculate the similarity of the input Sis_validsmilesMILES to property molecule
        similarity = round(DataStructs.FingerprintSimilarity(prop_mol_fsp, smiles_fsp), 4)
    else:
        similarity = -1     

    return similarity


def calculate_sascore(smiles: str):
    try: 
        mol = Chem.MolFromSmiles(smiles)

        # Calculate SAScorescore
        return round(sascorer.calculateScore(mol), 4)
    except:
        return -1
    

def get_smiles_stats(smiles_arr: list) -> dict:
    valid_smiles_arr, non_valid_smiles_arr = [], []
    canon_smiles_arr, non_canon_smiles_arr, converted_canon_smiles_arr = [], [], []
    
    for smiles in tqdm(smiles_arr):
        is_validsmiles = check_validness(smiles)
        canon_smiles=""
        
        if is_validsmiles:
            valid_smiles_arr.append(smiles)

            is_canonsmiles = check_canonical(smiles)

            if is_canonsmiles:
                canon_smiles_arr.append(smiles)
            else:
                non_canon_smiles_arr.append(smiles)
                # convert to canonical
                canon_smiles = get_canonical_form(smiles)
                converted_canon_smiles_arr.append(canon_smiles) 

        else:
            non_valid_smiles_arr.append(smiles)
            print("Non-valid smiles", smiles)        
    
    stats = {
        "valid": {"arr": valid_smiles_arr},
        "non_valid": {"arr": non_valid_smiles_arr},
        "canonical": {"arr": canon_smiles_arr, "converted_arr": converted_canon_smiles_arr},
        "non_canonical": {"arr": non_canon_smiles_arr},
    }
    
    return stats     