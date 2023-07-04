import time
from tqdm import tqdm
import parmap

from utils.db_utils import db_connect, make_column_indexed, table_row_count, add_column, get_chunck, update_chunck
from utils.chem_utils import get_selfies_form


def get_value_template(row):
    id, smiles = row
    prop_val = prop_fn(smiles) 

    return prop_val, id



if __name__ == "__main__":
    table_name = "GDB13"
    col_name = "selfies"
    col_type = "TEXT"
    prop_fn = get_selfies_form
    chunk_size = 1_000_000
    n_workers = 3
    t0 = time.time() 

    # Connect to GDB13 database
    con, cur = db_connect()
    

    # Rows count
    t0 = time.time() 
    total_rows = table_row_count(cur, table_name)
    print(f"Table row count is {total_rows}, Time [{time.time() - t0:.1f} sec]")

    # Add new column 'name' indexed
    # t1 = time.time() 
    # add_column(con, cur, table_name, col_name, type=col_type, index=False)
    # print(f"Time [{time.time() - t1:.1f} sec]")


    print("Inserting row values ...")

    for i in tqdm(range(0, total_rows, chunk_size)):
        # Execute a query to fetch the next chunk of data
        rows = get_chunck(cur, table_name, chunk_size, i)
        
        # Calculate new column values
        update_value = parmap.map(get_value_template, rows, pm_processes=n_workers)

        # Set values to the table
        update_chunck(con, cur, table_name, col_name, update_value)

    # Making column indexed
    make_column_indexed(con, cur, table_name, col_name, show_time=True) 

    print("Time", time.time() - t0)
    con.close()   