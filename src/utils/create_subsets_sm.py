import os
import sys
import random
import time
import json
import sqlite3
from tqdm import tqdm
import selfies as sf

#The path of the databases
GDB13_FILE_PATH = '/mnt/2tb/chem/GDB13.db'


def save_subset(rows, file_path):
    print("Saving the subset to JSONL ...")
    with open(file_path, "w", encoding="utf-8") as f:
        for row in tqdm(rows):
            json.dump({"text": sf.encoder(row[0])}, f)
            f.write("\n")


def get_subset(condition, chunk_size = 1_000_000):
    rows_list = []
    for i in tqdm(range(0, 977468301, chunk_size)):
        cursor.execute(condition, (i,i+chunk_size))
        rows = cursor.fetchall() 
        rows_list = rows_list + rows 

    random.Random(seed).shuffle(rows_list)
    return rows_list


if __name__ == "__main__":
    select_eqdist = 'SELECT name FROM GDB13 WHERE drug_sim>=0.20 AND drug_sim<0.2165 AND poison_sim>=0.20 AND poison_sim<0.2165 AND rowid > ? AND rowid <= ?;'
    select_druglike = 'SELECT name FROM GDB13 WHERE drug_sim>0.4 AND poison_sim<=0.4 AND rowid > ? AND rowid <= ?;'
    select_sas = 'SELECT name FROM GDB13 WHERE sascore<=3 AND rowid > ? AND rowid <= ?;'
    select_aspirin = 'SELECT name FROM GDB13 WHERE aspirin_similarity>=0.4 AND rowid > ? AND rowid <= ?;'

    aspirin_path = '/mnt/2tb/chem/hasmik/GDB_Generation_project/Molecular_Generation_with_GDB13/data/aspirin_0.4_sf/aspirin_0.4_sf.jsonl'
    sas_path = '/mnt/2tb/chem/hasmik/GDB_Generation_project/Molecular_Generation_with_GDB13/data/sas_3_sf/sas_3_sf.jsonl'
    druglike_path = '/mnt/2tb/chem/hasmik/GDB_Generation_project/Molecular_Generation_with_GDB13/data/druglike_0.4_sf/druglike_0.4_sf.jsonl'
    eqdist_path = '/mnt/2tb/chem/hasmik/GDB_Generation_project/Molecular_Generation_with_GDB13/data/eqdist_0.2_sf/eqdist_0.2_sf.jsonl'
    
    subset_paths = [aspirin_path, sas_path, druglike_path, eqdist_path]
    conditions = [select_aspirin, select_sas, select_druglike, select_eqdist]

    try:
        conn = sqlite3.connect(GDB13_FILE_PATH)
        cursor = conn.cursor()
    except:
        print(f"Error opening {GDB13_FILE_PATH} file")
        sys.exit(1)
    seed = 1    

    for condition, subset_path in zip(conditions, subset_paths):
        time1 = time.time()
        subset_rows = get_subset(condition=condition)
        print("Get subset", time.time() - time1)

        time2 = time.time()
        save_subset(subset_rows, subset_path)
        print("Save subset", time.time()-time2)

    conn.close()   