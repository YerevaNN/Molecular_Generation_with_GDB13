import os
import sys
import glob
import csv
import time
import json
import openpyxl
import sqlite3
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from collections import OrderedDict
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import parmap
import selfies as sf

# rdkit warnings
import rdkit.RDLogger as rkl
import rdkit.rdBase as rkrb
rkrb.DisableLog('rdApp.error')
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from utils.chem_utils import check_canonical, check_validness


def read_file(filename: str) -> list:
    with open(filename, "r", encoding="utf-8") as f:
        data_arr = f.read().splitlines()
    
    return data_arr

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"

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


def get_smiles_stats(smiles_arr: list) -> dict:
    valid_count, valid_smiles_arr = 0, []
    canon_count, canon_smiles_arr = 0, []
    
    for smiles in tqdm(smiles_arr):
        is_validsmiles = check_validness(smiles)
        is_canonsmiles = check_canonical(smiles)

        valid_count += is_validsmiles
        canon_count += is_canonsmiles

        if is_validsmiles:
            valid_smiles_arr.append(smiles)

            if is_canonsmiles:
                canon_smiles_arr.append(smiles)

    # unique
    valid_unique_count = len(set(valid_smiles_arr))
    canon_unique_count = len(set(canon_smiles_arr))
    
    stats = {
        "valid": {"all": valid_count, "unique": valid_unique_count},
        "canonical": {"all": canon_count, "unique": canon_unique_count}
    }
    
    return stats


def create_table(table_name: str):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    result = cursor.fetchall() 

    if table_name not in list(map(lambda a: a[0], result)):
        cursor.execute(f'''CREATE TABLE IF NOT EXISTS '{table_name}'
            (id INTEGER PRIMARY KEY     AUTOINCREMENT,
            name           TEXT    NOT NULL);''')
        
        cursor.execute(f'''CREATE INDEX 'canonical_index_{table_name}' ON '{table_name}' (name);''')

        conn.commit()
        

def load_data_into_table(cursor, file_path: str, table_name: str, repr: str = ""):
    cursor.execute(f''' SELECT * FROM '{table_name}' ''')
    result = cursor.fetchone()
    
    if result is None:
        print("Creating a table from", file_path)
        _, ext = os.path.splitext(file_path)
 
        if ext == ".jsonl":
            with open(file_path, 'r') as f:
                for row in tqdm(f):
                    line = json.loads(row)
                    smiles = line["text"] 

                    if repr == "selfies":
                        try: 
                            smiles = sf.decoder(smiles) 
                            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                        except:
                            continue

                    cursor.execute(f"INSERT INTO '{table_name}' (name) VALUES (?)", (smiles,))

    
        conn.commit()  


def load_generation_into_table(cursor, smiles_arr, table_name: str, repr: str = ""):
    cursor.execute(f''' SELECT * FROM '{table_name}' ''')
    result = cursor.fetchone()

    chunck_size = 1_000_000

    if result is None:
        smiles_arr_copy = list(smiles_arr)
        for i in tqdm(range(0,len(smiles_arr), chunck_size)):
            cursor.executemany(f"INSERT INTO '{table_name}' (name) VALUES (?)", [(s,) for s in smiles_arr_copy[i: i + chunck_size]])

            conn.commit() 


def calc_stats(smi):

    try:
        # make canonical
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        return smi
    except:
        pass            

    

def delete_tables():
    # select all temporary tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [t[0] for t in cursor.fetchall() if t[0] not in ["GDB13", "sqlite_sequence"]] 

    for name in table_names:
        if "train" not in name:
            cursor.execute(f"DROP TABLE IF EXISTS '{name}';")

    conn.commit()
    print("Temporary tables are deleted.")    


if __name__ == "__main__":
    # subspace_dict = {
    #     "sas": {
    #         "folder_substr": "sas_3_selfies",
    #         "column_name": "sascore",
    #         "thresh": "3",
    #         "sign": "<="
    #     },
    #     "aspirin": {
    #         "folder_substr": "aspirin_0.4_selfies",
    #         "column_name": "aspirin_similarity",
    #         "thresh": "0.4",
    #         "sign": ">="
    #     }
    # }
    # subspace = subspace_dict["aspirin"]

    # sampling = "top" # temp / top
    repr = "selfies" # smiles / selfies
    folder_substr_train = "aspirin_0.4" # sas_3 / aspirin_0.4
    
    
    
    column_name = "aspirin_similarity" #aspirin_similarity/sascore
    thresh = "0.4" # 0.4 / 3
    sign = ">=" # >= / <=
    generated_count = 10_000_000
    chunk_size = 1024
    canon_path = "./canonical_saved.csv"
    n_workers = 1

    folder_substr = f"aspirin_0.4_sf_all_sizes_{generated_count}" # sas_3 / aspirin_0.4
    output_file = f'./ablations/statistics/Sampling_results_{folder_substr}.xlsx'
    # canon_path = "./Generations_sas_3_selfies/sample_2M/canonical_40M.csv"

    DB_FILE_PATH = '/mnt/xtb/knarik/Smiles.db'
    prop_json_file = "./scores_test.json"
    GDB13_count = 975820227
    start_time = time.time()

    # Connect to DB
    try:
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()
    except:
        print(f"Error opening {DB_FILE_PATH} file")
        sys.exit(1) 

    # if os.path.exists(prop_json_file):
    #     with open(prop_json_file) as json_file:
    #         prop_dict = json.load(json_file)   
    # else:
    #     prop_dict = {}  


    # Delete previous tables
    delete_tables()      

    # Create Train table if not exists
    # create_table(f"train_{folder_substr_train}")

    # Load Train data if not exists
    # load_data_into_table(cursor, f"data/data_subsets/aspirin_0.4/train_aspirin_0.4_sm.jsonl", f"train_{folder_substr_train}", repr=repr) 


    if os.path.exists(output_file):
            df = pd.read_excel(output_file)
            print(df)
    else:
        # Default model
        df = pd.DataFrame(
            columns=[
                "Model",    
                "Non Valid Smiles", 
                "Canonical Smiles Unique",
                "In_Subset Unique",
                "In_GDB13 Unique",
                "In_GDB13 & Subset Unique",
                "Train Recall Unique",
                "Generated Smiles count"
                ]
            ) 


    # if os.path.exists(prop_json_file):
    #     with open(prop_json_file) as json_file:
    #         prop_dict = json.load(json_file)   
    # else:
    #     prop_dict = {}              
        
    # Get generation file paths
    generated_smiles_paths = sorted(glob.glob(f'./ablations/generations/*.csv'))


    for smiles_path in tqdm(generated_smiles_paths):
        last_part = os.path.basename(smiles_path.split("/")[-1])
        model_name = smiles_path.split("/")[-1].split(".csv")[0]

        if f'{model_name}' in df["Model"].values:
            continue

        print(f"================== Reading from {smiles_path} ====================")
        smiles_arr = read_file(f"./{smiles_path}")       

        # non-empty 
        smiles_arr = [s for s in smiles_arr if s][:generated_count]
        smiles_real_count = len(smiles_arr)
        print("Smiles count", smiles_real_count)

        # if smiles_real_count < generated_count:
        #     print(f"Generated smiles count is less than {generated_count}")
        #     continue

        print(f"================== Calculating valid/canonical counts ====================")    

        canon_smiles_arr = []
        non_valid_count = 0

        
        # if os.path.exists(canon_path):
        #     canon_smiles_arr = read_file(canon_path)  
        # else:
        #     canon_smiles_arr = parmap.map(calc_stats, smiles_arr, pm_processes=n_workers, pm_pbar=True)
        canon_smiles_arr = smiles_arr    
        # make unique canonical
        canon_smiles_arr = set(canon_smiles_arr) 
        canon_smiles_count = len(canon_smiles_arr)   

        # with open(canon_path, "w") as f:
        #     for smi in tqdm(canon_smiles_arr):
        #         if smi:
        #             f.write(smi+"\n" )
        

        print(f"================== Loading Sample to db ====================")
        time_1 = time.time()
        create_table(model_name)
        load_generation_into_table(cursor, canon_smiles_arr, model_name, repr=repr)
        print(time.time() - time_1, "sec")

        if column_name:
            print(f"================== In subset ====================")
            time_2 = time.time()
            print("Joining with GDB13 to get subset values...")
            cursor.execute(f'''SELECT '{model_name}'.name, GDB13.{column_name}
                            FROM '{model_name}' 
                            JOIN GDB13
                            ON '{model_name}'.name = GDB13.name
                            WHERE GDB13.{column_name} {sign} {thresh}''')
            
            subset_fromGDB_arr = cursor.fetchall()
            print("Time", (time.time() - time_2), "sec")
            
            print(f"================== In subset count manually ====================")
            # count of the subset that was in GDB
            time_3 = time.time()
            subset_GDB_inter = len(subset_fromGDB_arr)
            in_subset_count = len(subset_fromGDB_arr) 
            canon_smiles_arr = set(canon_smiles_arr)

            subset_nonGDB_arr = canon_smiles_arr - set(map(lambda a: a[0], subset_fromGDB_arr))
            scores = list(map(lambda a: a[1], subset_fromGDB_arr))

            for smi in tqdm(subset_nonGDB_arr):
                try:
                    mol = Chem.MolFromSmiles(smi)

                    if column_name == "sascore":
                        # Calculate SAScore
                        score = round(sascorer.calculateScore(mol), 3)

                    elif column_name == "aspirin_similarity": 
                        score = calculate_similarity(smi, ASPIRIN_SMILES)
                    else:
                        score = None    

                    # scores.append(score)    

                    if eval(f"{score} {sign} {thresh}"):
                        in_subset_count += 1         
                except:
                    pass


            # prop_dict[smiles_path] = scores  
            # print("Props stats saved at", prop_json_file)
            # with open(prop_json_file, "w") as f:
            #     json.dump(prop_dict, f)    
                
            
            print("Time", (time.time() - time_3), "sec")


        print(f"================== In GDB13 ====================")
        # Join GDB13 and generated smiles
        time_2 = time.time()
        print("Joining with GDB13 ...")
        cursor.execute(f'''SELECT COUNT(GDB13.name)
                        FROM GDB13
                        JOIN '{model_name}' 
                        ON GDB13.name = '{model_name}'.name''')
        
        in_GDB_unique = cursor.fetchone()[0]
        print("Time", (time.time() - time_2) , "sec")


        print(f"================== Train Recall ====================")    
        time_3 = time.time()

        print("Joining with Train where ...")
        cursor.execute(f'''SELECT COUNT('train_{folder_substr_train}'.name)
                        FROM 'train_{folder_substr_train}'
                        JOIN '{model_name}' 
                        ON 'train_{folder_substr_train}'.name = '{model_name}'.name''')
        
        from_train_unique = cursor.fetchone()[0]
        print("Time", (time.time() - time_3), "sec")

        # Add as row
        df.loc[len(df.index)] = [
            model_name,
            non_valid_count,
            canon_smiles_count,
            in_subset_count if column_name else "-",
            in_GDB_unique,
            subset_GDB_inter if column_name else "-",
            from_train_unique,
            smiles_real_count
        ]

        # Save
        df.to_excel(output_file, index=False)

        
    print("For one sample stat", (time.time() - start_time) / 60, "min")
    conn.close()    
