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

from utils.db_utils import db_connect
from utils.chem_utils import ASPIRIN_SMILES, calculate_similarity, calculate_sascore


if __name__ == "__main__":
    table_name = "GDB13"
    chunk_size = 1_000_000
    n_workers = 1
    t0 = time.time() 

    # Connect to GDB13 database
    con, cur = db_connect()
    
    def calculate_prop(row):
        id, smiles, _, _ = row
        similarity = None
        score = None

        # Calculate the similarity of the input SMILES to aspirin
        similarity = calculate_similarity(smiles, ASPIRIN_SMILES)

        # Calculate SAScorescore
        score = calculate_sascore(smiles)

        return similarity, score, id


    total_rows = 977468301
    chunk_size = 1_000_000
    n_workers = 1

    t1 = time.time()
    for i in tqdm(range(0, total_rows, chunk_size)):
        
        # Execute a query to fetch the next chunk of data
        cur.execute('SELECT * FROM GDB13 LIMIT ? OFFSET ?', (chunk_size, i))
        rows = cur.fetchall()
        
        # Calculate 2 column values
        update_values = parmap.map(calculate_prop, rows, pm_processes=n_workers)

        # Set Values to the table
        update_query = 'UPDATE GDB13 SET aspirin_similarity = ?, sascore = ? WHERE id = ?'
        cur.executemany(update_query, update_values)
        con.commit()


    print(time.time() - t1)

    
    con.close()   
