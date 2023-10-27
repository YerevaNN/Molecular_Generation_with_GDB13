import os
import sys
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import selfies as sf

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
        # Calculate the similarity of the input SMILES to property molecule
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