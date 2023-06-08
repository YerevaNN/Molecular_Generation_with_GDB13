import os
import sys
import glob
import csv
import time
import openpyxl
import sqlite3
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import RDConfig
import parmap
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


DB_FILE_PATH = '/mnt/xtb/knarik/Smiles.db'


if __name__ == "__main__":
    # Connect to DB
    try:
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()
    except:
        print(f"Error opening {DB_FILE_PATH} file")
        sys.exit(1)

    # Generate the MACCS keys for aspirin
    aspirin_mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')
    aspirin_fsp = MACCSkeys.GenMACCSKeys(aspirin_mol)
    
    
    def calculate_prop(row):
        id, smiles, _, _ = row
        similarity = None
        score = None

        try:
            # Convert the input SMILES to a RDKit molecule
            mol = Chem.MolFromSmiles(smiles)

            # Retrieve fingerprint
            smiles_fsp = MACCSkeys.GenMACCSKeys(mol)

            # Calculate the similarity of the input SMILES to aspirin
            similarity = round(DataStructs.FingerprintSimilarity(aspirin_fsp, smiles_fsp), 4)

            # Calculate SAScorescore
            score = round(sascorer.calculateScore(mol), 4)
        except:
            pass    

        return similarity, score, id


    total_rows = 977468301
    chunk_size = 1_000_000
    n_workers = 1

    t1 = time.time()
    for i in tqdm(range(968_000_001, total_rows, chunk_size)):
        
        # Execute a query to fetch the next chunk of data
        cursor.execute('SELECT * FROM GDB13 LIMIT ? OFFSET ?', (chunk_size, i))
        rows = cursor.fetchall()
        
        # Calculate 2 column values
        update_values = parmap.map(calculate_prop, rows, pm_processes=n_workers)

        # Set Values to the table
        update_query = 'UPDATE GDB13 SET aspirin_similarity = ?, sascore = ? WHERE id = ?'
        cursor.executemany(update_query, update_values)
        conn.commit()


    print(time.time() - t1)

    
    conn.close()   
