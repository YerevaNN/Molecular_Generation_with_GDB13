import json
from tqdm import tqdm
from utils.db_utils import db_connect


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
        

def load_data_into_table(conn, cursor, file_path: str, table_name: str):
    cursor.execute(f''' SELECT * FROM '{table_name}' ''')
    result = cursor.fetchone()
    
    if result is None:
        print("Creating a table from", file_path)

        with open(file_path, 'r') as f:
            for row in tqdm(f):
                line = json.loads(row)
                smiles = line["text"] 

                cursor.execute(f"INSERT INTO '{table_name}' (name) VALUES (?)", (smiles,))

        conn.commit()  


if __name__ == "__main__":
    # Connect to DB
    conn, cursor = db_connect("/mnt/xtb/knarik/Smiles.db")
    subset = "sas_3"

    # Create Train table if doesn't exist
    create_table(conn, cursor, f"train_{subset}")

    # Load Train data if doesn't exist
    load_data_into_table(conn, cursor, f"./data/data_bin_{subset}_sm_1000K/train/00/train_{subset}_sm.jsonl", f"train_{subset}") 
