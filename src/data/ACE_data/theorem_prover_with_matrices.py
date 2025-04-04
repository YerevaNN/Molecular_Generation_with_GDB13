#!/usr/bin/env python3

import logging
import time
import argparse
import subprocess
import numpy as np
import os
import glob
from tqdm import tqdm
import concurrent.futures
import tempfile
from pathlib import Path
import threading
from scipy.sparse import csr_matrix, save_npz, load_npz
from bitarray import bitarray

import cProfile
import pstats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# # Create logger
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

# # Info handler - will go to .out file
# info_handler = logging.StreamHandler()
# info_handler.setLevel(logging.INFO)
# info_handler.addFilter(lambda record: record.levelno <= logging.INFO)
# logger.addHandler(info_handler)

# # Error handler - will go to stderr, which will go to .err file
# error_handler = logging.StreamHandler()
# error_handler.setLevel(logging.ERROR)
# logger.addHandler(error_handler)



def run_eprover(axiom_line, conj_line, eprover_path, timeout=10):
    """
    Creates a temporary .tptp file with the given axiom and conjecture lines,
    runs E Prover with a timeout, and parses the output.

    Returns:
      1 if "SZS status Theorem" is found
      0 if "SZS status CounterSatisfiable" is found
     -1 otherwise (including timeout or unexpected output)
    """

    # Write axiom and conjecture to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tptp', delete=False) as tmp_f:
        temp_filename = tmp_f.name
        tmp_f.write(axiom_line.rstrip("\n") + "\n")
        tmp_f.write(conj_line.rstrip("\n") + "\n")

    # Construct the command to call E Prover
    cmd = [eprover_path, temp_filename, "--auto", "-s"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        output = result.stdout
    except subprocess.TimeoutExpired:
        # Timed out
        logging.error(f"[TIMEOUT] E Prover timed out for file: {temp_filename}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return -1
    except Exception as e:
        logging.error(f"[ERROR] E Prover call failed for file {temp_filename}: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return -1

    # Cleanup the temp file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    # Check the output for SZS statuses
    if "SZS status Theorem" in output:
        return 1
    elif "SZS status CounterSatisfiable" in output:
        return 0
    else:
        logging.error(f"Unexpected E Prover output: {output}")
        return -1


############################################################
# Function to call Vampire on a single (axiom, conjecture)
############################################################
def run_vampire(axiom_line, conj_line, vampire_path, timeout=10):
    """
    Creates a temporary .tptp file with the given axiom and conjecture lines,
    runs Vampire with a timeout, and parses the output.

    Returns:
      1 if "SZS status Theorem" is found
      0 if "SZS status CounterSatisfiable" is found
     -1 otherwise (including timeout or unexpected output)
    """
    # Each worker writes a unique temporary file to avoid collisions
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tptp', delete=False) as tmp_f:
        temp_filename = tmp_f.name
        tmp_f.write(axiom_line.rstrip("\n") + "\n")
        tmp_f.write(conj_line.rstrip("\n") + "\n")

    cmd = [vampire_path, temp_filename]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        output = result.stdout
        # If the process had non-zero return code, we still parse the output.
    except subprocess.TimeoutExpired:
        # Timed out
        logging.error(f"[TIMEOUT] Vampire timed out for file: {temp_filename}")
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return -1
    except Exception as e:
        # Some other error, e.g. OSError
        logging.error(f"[ERROR] Vampire call failed for file {temp_filename}: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return -1

    # Cleanup the temp file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    # Check the output for SZS statuses
    if "SZS status Theorem" in output:
        return 1
    elif "SZS status CounterSatisfiable" in output:
        return 0
    else:
        logging.error(f"Unexpected Vampire output: {output}")
        return -1
    



def worker_fn(task, prover_path, timeout):
    """Top-level function for parallel calls."""
    row, col, ax_line, conj_line = task

    if "vampire" in prover_path:
        res = run_vampire(ax_line, conj_line, prover_path, timeout=timeout)
    elif "eprover" in prover_path:
        res = run_eprover(ax_line, conj_line, prover_path, timeout=timeout)    
    else:
        raise NameError("There is no such a prover")    
    return (row, col, res)


def load_tptp(input_path):
    time_start = time.time()

    with open(input_path, "r", encoding="utf-8") as f:
        axioms = []
        conjs = []
        print(f"Reading {input_path} ...")

        for line in tqdm(f, leave=False):
            line = line.strip()

            if line:
                axioms.append(line)

                # Precompute the conj line for each formula
                conjs.append(line.replace(", axiom,", ", conjecture,"))

        print(f"Finished. Count of axioms is {len(axioms)}. Time {time.time() - time_start:.3f} sec.")   

    return axioms, conjs         



def process_chunk(start, end, axioms, conjs, prover_path, workers, timeout, input_path, output_path):
    """Processes a single chunk of the matrix."""
    N = end - start
        
    # NxN matrix
    matrix = np.full((N, N), 2, dtype=np.int8)
    np.fill_diagonal(matrix, 1)

    print(f"Matrix is {N}x{N}")

    base_name = os.path.basename(input_path)
    
    # profiler = cProfile.Profile()
    # profiler.enable()

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # Get all first rows and first columns
        for n in tqdm(range(N)):
            # Extract row n (is the same as column n)
            row_n = matrix[n, :] 

            # Find indices in row_n / col_n where value is not 1 or 0
            indexes = np.where((row_n != 1) & (row_n != 0))[0].tolist()

            if len(indexes) == 0:
                #logging.info("Continue")
                continue

            current_tasks = ([(n, idx, axioms[n], conjs[idx]) for idx in indexes] +
                             [(idx, n, axioms[idx], conjs[n]) for idx in indexes])       

            futures = [
                executor.submit(worker_fn, t, prover_path, timeout)
                for t in current_tasks
            ]

            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {base_name}", leave=False):
                row, col, res_val = fut.result()
                matrix[row, col] = res_val

                if res_val == -1:
                    logging.error(f"Unexpected value ({row}, {col}) start={start}")

            # Mask for equivalent elements
            mask_for_equiv = (matrix[n, :] == 1) & (matrix[:, n] == 1)

            # Find out which elements are equivalent in row/col
            matrix[n] = matrix[n] * mask_for_equiv
            matrix[:, n] = matrix[n]

            # Find all equivalent elements in whole matrix
            matrix[np.ix_(mask_for_equiv, mask_for_equiv)] = 1

            # Mask for non equivalent elements
            mask_for_non_equiv = ~mask_for_equiv

            matrix[np.ix_(mask_for_non_equiv, mask_for_equiv)] = 0
            matrix[np.ix_(mask_for_equiv, mask_for_non_equiv)] = 0

    # logging.info(matrix)        
    # profiler.disable()

    # stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")

    # stats.print_stats(20)
    equiv_f_name = f"{Path(input_path).stem}_chunk_{start}_{end}.npz"
    equiv_out_path = os.path.join(output_path, equiv_f_name)

    # Save to scipy sparse matrix
    sci_matrix = csr_matrix(matrix)
    save_npz(equiv_out_path, sci_matrix)

    # np.save(equiv_out_path, matrix)
    print(f"Saved {N}x{N} matrix -> {equiv_out_path}")


############################################################
# Function to process a single .tptp file
############################################################
def process_tptp_file(chunk_size, input_path, output_path, prover_path, workers, num_threads, timeout):
    """
    Reads the TPTP formulas from 'input_path' (one formula per line, each labeled 'axiom'),
    builds an NxN matrix of entailment results by calling Vampire, and saves to 'output_path'.

    Optimizations:
      - We only do j in range(i, N) to avoid duplicate checks.
      - We fill matrix[i,i] = 1 (any formula entails itself).
      - We precompute the conj lines.
      - We run calls in parallel (ProcessPoolExecutor) with 'workers'.
      - We add timeouts and handle concurrency carefully with NamedTemporaryFile.
    """
    # Load tptp files
    axioms, conjs = load_tptp(input_path)

    axioms_len = len(axioms)
    total_calls = axioms_len * axioms_len

    print(f"Total calls to be made: {total_calls}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as thread_executor:
        futures = []

        for start in range(0, axioms_len, chunk_size):
            end = min(start + chunk_size, axioms_len)
            axioms_N = axioms[start:end]
            conjs_N = conjs[start:end]

            futures.append(thread_executor.submit(
                process_chunk, start, end, axioms_N, conjs_N, prover_path, workers, timeout, input_path, output_path
            ))

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing chunks"):
            fut.result()
        

############################################################
# Main script: iterate over all .tptp files in a folder
############################################################
def main():
    parser = argparse.ArgumentParser(
        description="Iterate over all .tptp files in a folder, run pairwise Vampire checks in parallel with timeouts, and store NxN results."
    )
    parser.add_argument("--input", required=True,
                        help="Path to a folder containing .tptp files.")
    parser.add_argument("--output", required=True,
                        help="Path to a folder for storing .npy result files.")
    parser.add_argument("--chunk_size", type=int,
                        help="Chunk size is the size of the matrices.")
    parser.add_argument("--prover_path", default="/auto/home/knarik/Molecular_Generation_with_GDB13/src/data/ACE_data/vampire",
                        help="Path to the Vampire/Eprover executable.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default=1).")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of parallel threads for chunking (default=1).")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Timeout in seconds for each Vampire call (default=50).")

    args = parser.parse_args()
    start_time = time.time()

    # Ensure output folder exists
    os.makedirs(args.output, exist_ok=True)

    # Gather all .tptp files from the input folder
    pattern = os.path.join(args.input, "*.tptp")
    tptp_files = glob.glob(pattern)

    if not tptp_files:
        print(f"No .tptp files found in {args.input}")
        return    

    tptp_path_old_0 = "/nfs/h100/raid/chem/cf_9/card-uses.tptp"
    tptp_path_old_1 = "/nfs/h100/raid/chem/cf_9/card-customer-inserts.tptp"

    tptp_path_408 = "/nfs/h100/raid/chem/cf_11/big-card-uses.tptp"
    tptp_path_8414520 = "/nfs/h100/raid/chem/cf_11/big-card-customer-inserts-small-uses.tptp"

    tptp_files = [tptp_path_8414520]

    # Outer loop over .tptp files
    for tptp_path in tqdm(tptp_files, desc="All TPTP files", leave=False):
        t1 = time.time()                                                                            

        process_tptp_file(args.chunk_size, tptp_path, args.output, args.prover_path, workers=args.workers, num_threads=args.num_threads, timeout=args.timeout)
       
        print(f"{tptp_path} exec {time.time() - t1}")
        exit()

    print(f"Done. Processed all .tptp files. Time {(time.time() - start_time) / 3600} hours")

if __name__ == "__main__":
    main()
