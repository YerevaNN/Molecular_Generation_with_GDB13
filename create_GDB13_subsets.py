import os
import sys
import random
import time
import json
import sqlite3
from tqdm import tqdm

DB_FILE_PATH = '/mnt/xtb/knarik/Smiles.db'


def print_count(column):
    # Get the smiles count which are similar to 
    cursor.execute(f'SELECT COUNT(*) FROM GDB13 WHERE {column["name"]} {column["sign"]} ?', (column["threshold"],))
    count = cursor.fetchone()[0]
    print(f'Count of smiles where {column["name"]} {column["sign"]} {column["threshold"]} is {count}')  


def save_subset(rows, file_path):
    # Save JSONL
    print("Saving the subset to JSONL ...")
    with open(file_path, "w", encoding="utf-8") as f:
        for row in tqdm(rows):
            json.dump({"text": row[0]}, f)
            f.write("\n")


def get_subset(column):
    # Fetch rows
    cursor.execute(f'SELECT name FROM GDB13 WHERE {column["name"]} {column["sign"]} ?', (column["threshold"],))
    rows = cursor.fetchall()  

    # Shuffle
    random.Random(seed).shuffle(rows)

    return rows



if __name__ == "__main__":
    # Connect to DB
    try:
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()
    except:
        print(f"Error opening {DB_FILE_PATH} file")
        sys.exit(1)
    

    seed = 1
    columns = {
        "aspirin": {
            "name": "aspirin_similarity",
            "threshold": "0.4",
            "sign": ">="
            },
        # "sas": {
        #     "name": "sascore",
        #     "threshold": 3,
        #     "sign": "<="
        #     },
        }    

    # Show counts
    # print_count(columns["aspirin"])
    # print_count(columns["sas"])

    # Save subsets
    for key in tqdm(columns.keys()):
        time1 = time.time()
        subset_rows = get_subset(columns[key])
        print("Get subset", time.time() - time1)
        
        train_rows, valid_rows = subset_rows[:1_000_000], subset_rows[1_000_000:1_100_000]

        for split, rows in [('train', train_rows), ('valid', valid_rows)]:
            file_path = f'./data-bin/data-subsets/{split}_{key}_{columns[key]["threshold"]}.jsonl'
            time2 = time.time()
            save_subset(rows, file_path)
            print("Save subset", time.time() - time2)

    
    conn.close()   
