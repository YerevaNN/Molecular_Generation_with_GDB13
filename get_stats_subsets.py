import os
import sys
import csv
import time
import argparse
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
import rdkit.RDLogger as rkl
import rdkit.rdBase as rkrb
rkrb.DisableLog('rdApp.error')
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from utils.chem_utils import calculate_similarity, get_smiles_stats
from utils.db_utils import db_connect


ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
DRUG_SMILES = "CC(=O)NC1=CC=C(C=C1)O"
POISON_SMILES = "COC1=CC=C(C=C1)[N+](=O)[O-]"
GDB13_count = 975820227


def read_file(filename: str) -> list:
    with open(filename, "r", encoding="utf-8") as f:
        data_arr = f.read().splitlines()
    
    return data_arr


def create_table(conn, cursor, table_name: str):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    result = cursor.fetchall() 

    if table_name not in list(map(lambda a: a[0], result)):
        cursor.execute(f'''CREATE TABLE IF NOT EXISTS '{table_name}'
            (id INTEGER PRIMARY KEY     AUTOINCREMENT,
            name           TEXT    NOT NULL);''')
        
        cursor.execute(f'''CREATE INDEX 'canonical_index_{table_name}' ON '{table_name}' (name);''')

        conn.commit()
        print(f"{table_name} is created.")
        


def load_generation_into_table(conn, cursor, smiles_arr, table_name: str):
    cursor.execute(f''' SELECT * FROM '{table_name}' ''')
    result = cursor.fetchone()

    chunck_size = 1_000_000

    if result is None:
        smiles_arr_copy = list(smiles_arr)
        for i in tqdm(range(0,len(smiles_arr), chunck_size)):
            cursor.executemany(f"INSERT INTO '{table_name}' (name) VALUES (?)", [(s,) for s in smiles_arr_copy[i: i + chunck_size]])

            conn.commit()     
        
    

def delete_tables(conn, cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [t[0] for t in cursor.fetchall() if t[0] not in ["GDB13", "sqlite_sequence"]] 

    for name in table_names:
        if "train" not in name:
            cursor.execute(f"DROP TABLE IF EXISTS '{name}';")

    conn.commit()
    print("Temporary tables are deleted.") 



def cli_main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to a csv file.")
    parser.add_argument("--train_folder", type=str, help="The train folder path to load the train data into db.")
    parser.add_argument("--column_name", type=str, help="Subset's column name in db.")
    parser.add_argument("--condition", type=str, help="The condition to get the subset.")
    parser.add_argument("--gen_len", type=int, help="The generated molecules count.")
    parser.add_argument("--output_file", type=str, help="A xlsx file to output.")
    parser.add_argument("--scores_output_file", type=str, help="A json file to output the scores.")
    parser.add_argument("--db_path", type=str, help="The path where the GDB table exists.")
    args = parser.parse_args()
    

    # Connect to DB
    conn, cursor = db_connect(args.db_path)

    # Delete temporary tables
    delete_tables(conn, cursor)

    if os.path.exists(args.output_file):
            df = pd.read_excel(args.output_file)
            print(df)
    else:
        # Default model
        df = pd.DataFrame(
            columns=[
                "Model",    
                "Valid Smiles", 
                "Valid Smiles unique", 
                "Canonical Smiles",
                "Canonical Smiles unique",
                "Non_canonical Smiles",
                "Non_canonical Smiles unique",
                "Converted canonical Smiles",
                "Converted canonical Smiles unique",
                "In_GDB13",
                "In_GDB13 unique",
                "Non_GDB13",
                "Non_GDB13 unique",
                "In_GDB13 and Condition",
                "In_GDB13 and Condition unique",
                "Non_GDB13 and Condition",
                "Non_GDB13 and Condition unique",
                "In_train",
                "In_train unique",
                "Generated Smiles count"
                ]
            ) 

    if "#" in args.condition:
        # temporary solution
        args.condition = args.condition.replace("#", " ")  

    model_name = args.file_path.split("/")[-1].split(".csv")[0]

    if f'{model_name}' in df["Model"].values:
        print("The statistics already exists.")
        return 

    print(f"================== Reading from {args.file_path} ====================")
    smiles_arr = read_file(args.file_path)       

    # non-empty 
    smiles_arr = [s for s in smiles_arr if s][:args.gen_len]
    smiles_total_count = len(smiles_arr)
    print("Smiles count", smiles_total_count)

    if smiles_total_count < args.gen_len:
        print(f"Generated smiles count in file is less than {args.gen_len}")
        return
    
    print(f"================== Calculating valid/canonical counts ====================")    
    validness_stats = get_smiles_stats(smiles_arr)
    all_canon_arr = validness_stats["canonical"]["arr"] + validness_stats["canonical"]["converted_arr"]

   
    print(f"================== Loading Sample to db ====================")
    time_1 = time.time()
    # with duplicates
    create_table(conn, cursor, model_name)
    load_generation_into_table(conn, cursor, all_canon_arr, model_name)
    print(time.time() - time_1, "sec")
    from_GDB_arr = []

    if args.column_name:
        print(f"================== From GDB13 ====================")
        # Join GDB13 and generated smiles
        time_2 = time.time()
        print("Joining with GDB13 ...")
        cursor.execute(f'''SELECT GDB13.name, {args.column_name}
                        FROM GDB13
                        JOIN '{model_name}' 
                        ON GDB13.name = '{model_name}'.name''')
        
        from_GDB_arr = cursor.fetchall()
        print("Time", (time.time() - time_2) , "sec")

        print(f"================== From non-GDB13 ====================")
        # Left Join generated smiles and GDB
        time_2 = time.time()
        print("Joining with GDB13 ...")
        cursor.execute(f'''SELECT '{model_name}'.name, {args.column_name}
                        FROM '{model_name}' 
                        LEFT JOIN GDB13
                        ON '{model_name}'.name = GDB13.name
                        WHERE GDB13.name IS NULL''')
        
        from_non_GDB_arr = cursor.fetchall()
        print("Time", (time.time() - time_2) , "sec")


        print(f"================== From GDB and condition ====================")
        time_2 = time.time()
        print("Joining with GDB13 to get the values satisfying the condition ...")
        cursor.execute(f'''SELECT '{model_name}'.name, {args.column_name}
                        FROM '{model_name}' 
                        JOIN GDB13
                        ON '{model_name}'.name = GDB13.name
                        WHERE {args.condition}''')
        
        from_GDB_and_cond_arr = cursor.fetchall()
        print("Time", (time.time() - time_2), "sec")


        print(f"================== Outside GDB and condition (counting manually) ====================")
        # count of the subset that is in GDB
        time_3 = time.time()
        from_non_GDB_and_cond_arr = []
        scores_arr = from_GDB_arr[:]
 
        for smi, _ in tqdm(from_non_GDB_arr):
            try:
                mol = Chem.MolFromSmiles(smi)
                if args.train_folder == "sas_3":
                    # Calculate SAScore
                    score = round(sascorer.calculateScore(mol), 3)

                    if score <= 3:
                        from_non_GDB_and_cond_arr.append(smi) 

                elif args.train_folder == "aspirin_0.4": 
                    score = calculate_similarity(smi, ASPIRIN_SMILES)
                    if score >= 0.4:
                        from_non_GDB_and_cond_arr.append(smi) 

                elif args.train_folder == "druglike_0.4": 
                    score_drug = calculate_similarity(smi, DRUG_SMILES)
                    score_pois = calculate_similarity(smi, POISON_SMILES)

                    if score_drug>0.4 and score_pois<=0.4:
                        from_non_GDB_and_cond_arr.append(smi) 

                elif args.train_folder == "equal_dist": 
                    score_drug = calculate_similarity(smi, DRUG_SMILES)
                    score_pois = calculate_similarity(smi, POISON_SMILES)

                    if score_drug>=0.20 and score_drug<0.2165 and score_pois>=0.20 and score_pois<0.2165:
                        from_non_GDB_and_cond_arr.append(smi)            

                else:
                    score = None    

                scores_arr.append((smi, score))                            
            except:
                pass

        print("Props stats saved at", args.scores_output_file)
        with open(args.scores_output_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(scores_arr)    
        
        print("Time", (time.time() - time_3), "sec")


    print(f"================== Train Recall ====================")    
    time_3 = time.time()

    print("Joining with Train")
    cursor.execute(f'''SELECT 'train_{args.train_folder}'.name
                    FROM 'train_{args.train_folder}'
                    JOIN '{model_name}' 
                    ON 'train_{args.train_folder}'.name = '{model_name}'.name''')
    
    from_train_arr = cursor.fetchall()
    print("Time", (time.time() - time_3), "sec")

    # Add as row
    df.loc[len(df.index)] = [
        model_name,
        len(validness_stats["valid"]["arr"]),
        len(set(validness_stats["valid"]["arr"])),
        len(validness_stats["canonical"]["arr"]),
        len(set(validness_stats["canonical"]["arr"])),
        len(validness_stats["non_canonical"]["arr"]),
        len(set(validness_stats["non_canonical"]["arr"])),
        len(validness_stats["canonical"]["converted_arr"]),
        len(set(validness_stats["canonical"]["converted_arr"])),
        len(from_GDB_arr) if from_GDB_arr else "",
        len(set(from_GDB_arr)) if from_GDB_arr else "",
        len(from_non_GDB_arr) if from_GDB_arr else "",
        len(set(from_non_GDB_arr)) if from_GDB_arr else "",
        len(from_GDB_and_cond_arr) if from_GDB_arr else "",
        len(set(from_GDB_and_cond_arr)) if from_GDB_arr else "",
        len(from_non_GDB_and_cond_arr) if from_GDB_arr else "",
        len(set(from_non_GDB_and_cond_arr)) if from_GDB_arr else "",
        len(from_train_arr),
        len(set(from_train_arr)),
        smiles_total_count
    ]

    # Save
    df.to_excel(args.output_file, index=False)
    conn.close() 

    print("For one sample stat", (time.time() - start_time), "sec")

if __name__ == "__main__":
    cli_main()