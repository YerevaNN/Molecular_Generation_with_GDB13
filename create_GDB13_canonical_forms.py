import sys
import time
import sqlite3
from rdkit import Chem
from tqdm import tqdm
import parmap


DB_FILE_PATH = '/mnt/xtb/knarik/Smiles.db'


def gen_canonical_form(row):
    id, smiles_original = row
    smiles_canonical = None

    try:
        # Convert to canonical SMILES
        smiles_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_original), canonical=True)
    except:
        pass    

    return smiles_canonical, id


if __name__ == "__main__":
    # Connect to DB
    try:
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()
    except:
        print(f"Error opening {DB_FILE_PATH} file")
        sys.exit(1)


    # check if GDB13 is canonicalized
    cursor.execute("SELECT name FROM pragma_table_info('GDB13') where name='name_original'")
    result = cursor.fetchone()

    t0 = time.time()
    cursor.execute("SELECT COUNT(id) FROM GDB13")
    total_rows = cursor.fetchone()[0]

    print(f"Table row count is {total_rows}, [{time.time() - t0:.1f} sec]")

    # if not
    if result is None:
        cursor.execute("ALTER TABLE GDB13 RENAME COLUMN 'name' TO 'name_original'")
        cursor.execute("ALTER TABLE GDB13 ADD COLUMN 'name' TEXT NULL")
        conn.commit()

        total_rows = 977468301
        chunk_size = 1_000_000
        n_workers = 12

        t1 = time.time()
        
        print("Updating row values ...")
        for i in tqdm(range(0, total_rows, chunk_size)):
            # Execute a query to fetch the next chunk of data
            cursor.execute("SELECT id, name_original FROM GDB13 LIMIT ? OFFSET ?", (chunk_size, i))
            rows = cursor.fetchall()
            
            # Calculate new column values
            update_value = parmap.map(gen_canonical_form, rows, pm_processes=n_workers)

            # Set Values to the table
            update_query = "UPDATE GDB13 SET name = ? WHERE id = ?"
            cursor.executemany(update_query, update_value)
            conn.commit()

        print(time.time() - t1)

    conn.close()   