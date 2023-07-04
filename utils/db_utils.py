import sys
import time
import csv
import sqlite3
from tqdm import tqdm


GDB13_FILE_PATH = '/mnt/2tb/chem/GDB13.db'


def db_connect(db_file_path: str = GDB13_FILE_PATH):
    try:
        print(f"Connecting to db {db_file_path}...")
        con = sqlite3.connect(db_file_path)
        cur = con.cursor()
    except:
        print(f"Error opening {db_file_path} file.")
        sys.exit(1)

    return con, cur  


def table_row_count(cur, table_name: str, show_time=True):
    t = time.time() 
    print("Calculating rows count...")

    cur.execute(f"SELECT COUNT(id) FROM '{table_name}'")
    result = cur.fetchone()[0]

    if show_time:
        print("Time", (time.time() - t), "sec")

    return result


def update_column_name(con, cur, table_name: str, old_name: str, new_name: str):
    print(f"Updating {old_name} column name to {new_name}...")

    cur.execute(f"ALTER TABLE '{table_name}' RENAME COLUMN '{old_name}' TO '{new_name}'")
    con.commit()


def make_column_indexed(con, cur, table_name: str, col_name: str, show_time=True):
    t = time.time() 
    print(f"Adding index to {col_name}...")

    cur.execute(f"CREATE INDEX '{col_name}_index' ON {table_name}({col_name})")
    con.commit()    

    if show_time:
        print("Time", (time.time() - t), "sec")
    

def add_column(con, cur, table_name: str, col_name: str, type: str = 'TEXT', index=False):
    print(f"Adding column {col_name}...")
    cur.execute(f"ALTER TABLE '{table_name}' ADD COLUMN '{col_name}' {type} NULL")

    if index:
        print(f"Also adding index...")
        cur.execute(f"CREATE INDEX '{col_name}_index' ON {table_name}({col_name})")

    con.commit() 
    


def get_chunck(cur, table_name: str, chunk_size: int, idx):
    cur.execute(f"SELECT id, name FROM '{table_name}' ORDER BY id LIMIT ? OFFSET ?", (chunk_size, idx))
    rows = cur.fetchall()   

    return rows 


def update_chunck(con, cur, table_name: str, col_name: str, update_value):
    cur.executemany(f"UPDATE '{table_name}' SET '{col_name}' = ? WHERE id = ?", update_value) 

    con.commit()    


def get_subspace_and_save(cur, table_name: str, col_name: str, where_st:str, output_path:str, show_time=True):
    t = time.time() 

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', col_name])    

        # Fetch rows
        cur.execute(f"SELECT name FROM '{table_name}' WHERE '{col_name}' ?", (where_st,))
        rows = cur.fetchall()  
        # Write rows
        writer.writerows(rows)    

    if show_time:                
        print("Time", (time.time() - t), "sec")     

    return rows        



