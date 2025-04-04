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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

############################################################
# Function to call Vampire on a single (axiom, conjecture)
###########################################################
def run_vampire(axiom_line, conj_line, vampire_path, timeout=50, attempt=0):
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
        logging.info(f"[TIMEOUT] Vampire timed out for file: {temp_filename}")
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return -1
    except Exception as e:
        # Some other error, e.g. OSError
        logging.info(f"[ERROR] Vampire call failed for file {temp_filename}: {e}")
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
        # logging.info(f"Unexpected Vampire output: `{output}` at attempt {attempt}")
        # if attempt == 20:
        #      return -1
        # return run_vampire(axiom_line, conj_line, vampire_path, timeout=50, attempt=attempt+1) 
        return -1

def worker_fn(task, vampire_path, timeout):
    """Top-level function for parallel calls."""
    row, col, ax_line, conj_line = task
    res = run_vampire(ax_line, conj_line, vampire_path=vampire_path, timeout=timeout)
    return (row, col, res)


############################################################
# Function to process a single .tptp file
############################################################
def process_tptp_file(input_path, output_path, vampire_path, workers=10, timeout=50):
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
    ##LOG
    logging.info(f"Reading lines from {input_path}")
    # Read lines
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in tqdm(f, desc=f"Reading {os.path.basename(input_path)}", leave=False)
                 if line.strip()]

    logging.info(f"Finished reading lines. We have {len(lines)} lines.")

    N = len(lines)

    # NxN matrix
    matrix = np.full((N, N), -1, dtype=int)

    # Precompute the conj_line for each formula
    conj_lines = [line.replace(", axiom,", ", conjecture,") for line in lines]

    # Build a list of tasks (i, j, axiom, conj) for A->B
    # plus (j, i, axiom, conj) for B->A, skipping i==j.
    # If i == j, matrix[i,i] = 1.

    tasks = []
    vampire_calls = 0

    for i in range(N):
        matrix[i, i] = 1  # self-entailment

        for j in range(i+1, N):
            # A->B
            tasks.append((i, j, lines[i], conj_lines[j]))
            # B->A
            tasks.append((j, i, lines[j], conj_lines[i]))

    total_calls = len(tasks)
    ##LOG
    logging.info(f"Total calls to be made: {total_calls}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        logging.info(f"Starting executor with {len(tasks)} tasks.")

        futures = [
            executor.submit(worker_fn, t, vampire_path, timeout)
            for t in tasks
        ]
        logging.info(f"Tasks are submitted.")

        for fut in tqdm(concurrent.futures.as_completed(futures), total=total_calls, desc=f"Processing {os.path.basename(input_path)}", leave=False):
            row, col, res_val = fut.result()
            matrix[row, col] = res_val
            vampire_calls += 1

    intersection = np.triu(matrix, k=1) & np.tril(matrix, k=-1).T

    # Add the diagonal back from the original matrix
    result = intersection + np.diag(np.diag(matrix))
    result[np.tril_indices_from(result, -1)] = result.T[np.tril_indices_from(result, -1)]
    print(result[:12, :12])


    logging.info(f"Saving matrix of shape {N}x{N} to {output_path}")
    # Save to .npy
    np.save(output_path, result)
    print(f"Saved {N}x{N} matrix -> {output_path}")

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
    parser.add_argument("--vampire", default="/auto/home/hrant/Recall-ACE/vampire",
                        help="Path to the Vampire executable.")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers (default=10).")
    parser.add_argument("--timeout", type=int, default=50,
                        help="Timeout in seconds for each Vampire call (default=50).")

    args = parser.parse_args()

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
    tptp_path_2 = "/nfs/h100/raid/chem/cf_9/card-uses.tptp"
    tptp_path_3 = "/nfs/h100/raid/chem/cf_9/card-customer-inserts.tptp"
    tptp_path_6 = "/nfs/h100/raid/chem/cf_9/big-card-customer-inserts-small-uses.tptp"

    tptp_files = [tptp_path_old_1]

    # Outer loop over .tptp files
    for tptp_path in tqdm(tptp_files, desc="All TPTP files", leave=False):
        base = os.path.splitext(os.path.basename(tptp_path))[0]
        out_npy_path = os.path.join(args.output, f"{base}.npy")
        t1 = time.time()
        
        process_tptp_file(tptp_path, out_npy_path, args.vampire, workers=args.workers, timeout=args.timeout)

        print(f"One task exec {time.time() - t1}")
        # print("One group is ready!!!")
        exit()

    print("Done. Processed all .tptp files.")

if __name__ == "__main__":
    main()
