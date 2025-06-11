import os
import glob
import time
import shutil
import logging
import argparse
import subprocess
import concurrent.futures
import threading
import aiofiles
import asyncio
from tqdm import tqdm
from pathlib import Path
from pprint import pformat

import psutil


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CPUUsageMonitor(threading.Thread):
    def __init__(self, interval=1):
        super().__init__()
        self.interval = interval
        self.running = True
        self.usage_sum = 0.0
        self.count = 0

    def run(self):
        while self.running:
            # psutil.cpu_percent(interval=1) returns a percentage over the interval for all cores
            usage = psutil.cpu_percent(interval=self.interval)
            self.usage_sum += usage
            self.count += 1

    def stop(self):
        self.running = False

    def get_average_usage(self):
        return self.usage_sum / self.count if self.count else 0.0


def log_args(args):
    args_dict = vars(args)
    print("Script Arguments:\n" + pformat(args_dict))
    

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

        # logging.info(f"Count of axioms is {len(axioms)}. Time {time.time() - time_start:.3f} sec.")   

    return axioms, conjs 


def load_tptp_from_clusters(input_paths):
    time_start = time.time()
    axioms = []
    conjs = []

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()

            if line:
                axioms.append(line)

                # Precompute the conj line for each formula
                conjs.append(line.replace(", axiom,", ", conjecture,"))
            else:
                print("something went wrong with line")   
                print(line)  

    # logging.info(f"Count of axioms is {len(axioms)}. Time {time.time() - time_start:.3f} sec.")   

    return axioms, conjs     


def load_txt(input_path):
    time_start = time.time()

    txt_path = str(Path(input_path).with_suffix("")) + ".txt"

    with open(txt_path, "r", encoding="utf-8") as f:
        txt_lines = []
        print(f"Reading {txt_path} ...")

        for line in tqdm(f, leave=False):
            line = line.strip()

            if line:
                txt_lines.append(line)

        # logging.info(f"Count of lines is {len(txt_lines)}. Time {time.time() - time_start:.3f} sec.")   

    return txt_lines 


def load_txt_from_clusters(input_paths):
    time_start = time.time()
    txt_lines = []

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()

            if line:
                txt_lines.append(line)

    # logging.info(f"Count of lines is {len(txt_lines)}. Time {time.time() - time_start:.3f} sec.")   

    return txt_lines 
        
    
def run_vampire(axiom_line, conj_line, vampire_path, show_progress, indx, timeout=10, attempt=0):
    """
    Runs Vampire with the given axiom and conjecture lines by piping them directly to its stdin.

    Returns:
      1 if "SZS status Theorem" is found
      0 if "SZS status CounterSatisfiable" is found
     -1 otherwise (including timeout or unexpected output)
    """
    # Combine the axiom and conjecture lines into a single input string
    vampire_input = f"{axiom_line.rstrip()}\n{conj_line.rstrip()}\n"

    # Define the command to run Vampire
    cmd = [vampire_path, '--input_syntax', 'tptp', '--mode', 'vampire']

    try:
        # Run the Vampire process with the input piped to its stdin
        result = subprocess.run(
            cmd,
            input=vampire_input,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        output = result.stdout
    except subprocess.TimeoutExpired:
        # Timed out
        logging.error("[TIMEOUT] Vampire timed out.")
        return -1
    except Exception as e:
        # Some other error
        logging.error(f"[ERROR] Vampire call failed: {e}")
        return -1

    # Check the output for SZS statuses
    if "SZS status Theorem" in output:
        return 1
    elif "SZS status CounterSatisfiable" in output:
        return 0
    else:
        if show_progress:
            logging.info(f"Unexpected Vampire output: `{output}` at attempt {attempt} at line {indx}.")

        if attempt == 3:
            return -1
        return run_vampire(axiom_line, conj_line, vampire_path, show_progress, indx, timeout=timeout, attempt=attempt + 1)


def worker_fn(task, prover_path, timeout, show_progress, indx):
    """Top-level function for parallel calls."""
    i_indx, ax_line, conj_line = task

    if "vampire" in prover_path:
        res = run_vampire(ax_line, conj_line, prover_path, show_progress, indx, timeout)
    else:
        raise NameError("There is no such a prover")    
    return (i_indx,  res)


def worker_fn_for_batch(task, prover_path, timeout, show_progress, indx):
    """Top-level function for parallel calls."""
    i_indx, j_indx, ax_line, conj_line = task

    if "vampire" in prover_path:
        res = run_vampire(ax_line, conj_line, prover_path, show_progress, indx, timeout)
    else:
        raise NameError("There is no such a prover")    
    return (i_indx, j_indx, res)


async def write_line_to_file(
    path: str,
    line: str,
    mode: str = "w",
    extension: str = "txt",
    retries: int = 0,
    max_retries: int = 3
) -> None:
    try:
        async with aiofiles.open(f"{path}.{extension}", mode) as f:
            await f.write(line + "\n")

    except Exception as e:
        if retries >= max_retries:
            raise OSError(f"Failed after {max_retries} retries: {e}")

        await write_line_to_file(path, line, mode, extension, retries + 1, max_retries)


async def write_to_files(cluster_path, txt_line, tptp_line, mode="w"):
    # write to txt file
    await write_line_to_file(cluster_path, txt_line, mode=mode, extension="txt"), 

    # write to tptp file
    if "conjecture" in tptp_line:
        tptp_line = tptp_line.replace(", conjecture,", ", axiom,")
        
    await write_line_to_file(cluster_path, tptp_line, mode=mode, extension="tptp")  


def merge_2_files(cluster_path, merge_file_path, extension="txt", retries=0, max_retries=3):  
    try:
        with open(f"{cluster_path}.{extension}", "a") as f_w, open(f"{merge_file_path}.{extension}", "r") as f_r:
            f_w.write(f_r.read())

    except Exception as e:
        if retries >= max_retries:
            raise OSError(f"Failed after {max_retries} retries: {e}")

        return merge_2_files(cluster_path, merge_file_path, extension, retries + 1, max_retries)  


async def merge_files(cluster_path, merge_file_path):
    # print(f"Merging file {cluster_path} and {merge_file_path}")

    # merge txt files
    merge_2_files(cluster_path, merge_file_path, extension="txt")

    # merge tptpt files
    merge_2_files(cluster_path, merge_file_path, extension="tptp")  


async def copy_files(new_path: str, old_path: str):
    """Asynchronously copy files and then remove the originals."""
    # Copy files
    await asyncio.gather(
        async_copy_file(f"{old_path}.txt", f"{new_path}.txt"),
        async_copy_file(f"{old_path}.tptp", f"{new_path}.tptp"),
    )


async def async_remove_file(path: str):
    """Asynchronously remove a file."""
    try:
        await asyncio.to_thread(os.remove, path)
    except Exception as e:
        print(f"Error deleting {path}: {e}")


async def async_copy_file(src: str, dst: str):
    """Asynchronously copy a file using a thread to avoid blocking the event loop."""
    await asyncio.to_thread(shutil.copy, src, dst)


def check_equivalence(axiom_line, conj_line, executor, prover_path, timeout, show_progress, indx):
    equivalence_result = [-1, -1]

    # change places of axiom and conj
    conj_line_to_axiom = conj_line.replace(", conjecture,", ", axiom,")
    axiom_line_to_conj = axiom_line.replace(", axiom,", ", conjecture,")

    current_pair = [[0, axiom_line, conj_line], [1, conj_line_to_axiom, axiom_line_to_conj]]     

    futures = [
        executor.submit(worker_fn, t, prover_path, timeout, show_progress, indx)
        for t in current_pair
    ]

    for fut in concurrent.futures.as_completed(futures):
        res_indx, res_val = fut.result()
        equivalence_result[res_indx] = res_val

        if res_val == -1 and show_progress:
            logging.error(f"Unexpected value for axiom ({axiom_line})")

    return equivalence_result  


def check_equivalence_batch(path_batch, axiom_line_batch, conj_line, executor, prover_path, timeout, show_progress, indx):
    current_pairs = []
    equivalence_results = {}

    # Change conj to be an axiom
    conj_line_to_axiom = conj_line.replace(", conjecture,", ", axiom,")

    for i in range(len(axiom_line_batch)):
        equivalence_results[-1, i] = -100
        equivalence_results[i, -1] = -100

        # Create pair: [row, col, axiom, conj] direction: treating axiom_line_batch[i] as axiom, conj_line as conjecture.
        current_pairs.append([-1, i, axiom_line_batch[i], conj_line])

        # Also create the reverse direction. Precompute the replacement for axiom_line_batch[i] once.
        axiom_line_to_conj = axiom_line_batch[i].replace(", axiom,", ", conjecture,")
        current_pairs.append([i, -1, conj_line_to_axiom, axiom_line_to_conj])   

    futures = [
        executor.submit(worker_fn_for_batch, t, prover_path, timeout, show_progress, indx)
        for t in current_pairs
    ]

    for fut in concurrent.futures.as_completed(futures):
        row, col, res_val = fut.result()
        equivalence_results[row, col] = res_val   

    # check whether there exist any equivalence inside batch
    contains_undefined_results = False 

    for i in range(len(axiom_line_batch)):
        if equivalence_results[-1, i] == equivalence_results[i, -1] == 1: 
            return 1, path_batch[i]
        elif equivalence_results[-1, i] == -1 or equivalence_results[i, -1] == -1: 
            contains_undefined_results = True

    if contains_undefined_results: 
        return -1, ""       
    else:
        return 0, ""
 

def check_equivalence_batch_optimized(path_batch, axiom_line_batch, conj_line, executor, prover_path, timeout, show_progress, indx):
    current_pairs_forward = []
    current_pairs_backward = []
    equivalence_results = {}

    # Change conj to be an axiom
    conj_line_to_axiom = conj_line.replace(", conjecture,", ", axiom,")

    for i in range(len(axiom_line_batch)):
        equivalence_results[-1, i] = -100
        equivalence_results[i, -1] = -100

        current_pairs_forward.append([-1, i, axiom_line_batch[i], conj_line])

    # Give only forward pairs
    futures = [
        executor.submit(worker_fn_for_batch, t, prover_path, timeout, show_progress, indx)
        for t in current_pairs_forward
    ]

    for fut in concurrent.futures.as_completed(futures):
        row, col, res_val = fut.result()
        equivalence_results[row, col] = res_val

        if res_val == 1:
            axiom_line_to_conj = axiom_line_batch[col].replace(", axiom,", ", conjecture,")
            current_pairs_backward.append([col, -1, conj_line_to_axiom, axiom_line_to_conj]) 

    # If any of forward pairs gives 1, then go check its backward pair
    if current_pairs_backward:
        futures = [
            executor.submit(worker_fn_for_batch, t, prover_path, timeout, show_progress, indx)
            for t in current_pairs_backward
        ]  

        for fut in concurrent.futures.as_completed(futures):
            row, col, res_val = fut.result()
            equivalence_results[row, col] = res_val      

        # check whether there exist any equivalence inside batch
        contains_undefined_results = False   
        
        for i in range(len(axiom_line_batch)):
            if equivalence_results[-1, i] == equivalence_results[i, -1] == 1: 
                return 1, path_batch[i]
            elif equivalence_results[-1, i] == -1 or equivalence_results[i, -1] == -1: 
                contains_undefined_results = True

        if contains_undefined_results: 
            return -1, ""       
        else:
            return 0, ""   
    else:
        return 0, "" 


def process_chunk(start, axioms_N, conjs_N, txt_N, prover_path, num_threads, input_path, output_path, timeout):
    asyncio.run(process_chunk_async(start, axioms_N, conjs_N, txt_N, prover_path, num_threads, input_path, output_path, timeout))


async def process_chunk_async(start, axioms_N, conjs_N, txt_N, prover_path, num_threads, input_path, output_path, timeout):
    t = time.time()
    line_count = len(axioms_N)
    
    # Each chunk creates a process_cluster for its own
    process_output_path = os.path.join(output_path, f"process_cluster_{start}")
    os.makedirs(process_output_path, exist_ok=True)
    
    # Create the first cluster file inside the process_cluster
    cluster_number = 0
    cluster_path = os.path.join(process_output_path, f"cluster_{cluster_number}")
    clusters = {cluster_path: axioms_N[0]}
    await write_to_files(cluster_path, txt_N[0], axioms_N[0], mode="w") 

    # Undefined files
    parent_path = os.path.dirname(output_path)
    undefined_path = os.path.join(parent_path, "undefined")

    if not os.path.exists(os.path.join(parent_path, "undefined.txt")):
        # Create a file for undefined (-1) equivalence results also
        await write_to_files(undefined_path, "", "", mode="w") 

    # For showing only one process output
    show_progress = start == 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as thread_executor:
        for indx, conj_line in tqdm(enumerate(conjs_N), disable=not show_progress):
            start_time = time.time()
            
            if indx == 0:
                continue
            
            belongs_to_a_cluster = False
            contains_undefined_results = False

            vampire_calls = 0

            for cluster_path, cluster_tptp in clusters.items():    
                equivalence_result = check_equivalence(cluster_tptp, conj_line, thread_executor, prover_path, timeout, show_progress, indx)
                vampire_calls += 1
                
                if equivalence_result[0] == equivalence_result[1] == 1:
                    belongs_to_a_cluster = True
                    # write to .txt and .tptp files
                    await write_to_files(cluster_path, txt_N[indx], conj_line, mode="a")
                    break

                if equivalence_result[0] == -1 or equivalence_result[1] == -1:
                    contains_undefined_results = True

            if not belongs_to_a_cluster and contains_undefined_results:
                # Write to a undefined results file
                await write_to_files(undefined_path, txt_N[indx], conj_line, mode="a")
            elif not belongs_to_a_cluster:
                # Create new cluster
                cluster_number += 1
                new_cluster_path = os.path.join(process_output_path, f"cluster_{cluster_number}")
                axiom_line = conj_line.replace(", conjecture,", ", axiom,")
                clusters[new_cluster_path] = axiom_line
                await write_to_files(cluster_path, txt_N[indx], axiom_line, mode="w")
            
            if show_progress:
                print(f"Running line {indx} / {line_count}, Duration: {(time.time() - start_time) / 60:.1f} min with {cluster_number} clusters, Vampire calls: {vampire_calls}, Processed: {(time.time() - t) / 3600:.1f} hours.")

def process_cluster_chunk(process_number, process_cluster_pair_0, process_cluster_pair_1, prover_path, num_threads, output_path, timeout):
    asyncio.run(process_cluster_chunk_async(process_number, process_cluster_pair_0, process_cluster_pair_1, prover_path, num_threads, output_path, timeout))


async def process_cluster_chunk_async(process_number, process_cluster_pair_0, process_cluster_pair_1, prover_path, num_threads, output_path, timeout):
    t = time.time()
    axioms_0, conjs_0, paths_0 = process_cluster_pair_0
    axioms_1, conjs_1, paths_1 = process_cluster_pair_1

    line_count_start = len(axioms_0)
    line_count_end = len(axioms_1)

    # Each chunk creates a process_cluster for its own
    process_output_path = os.path.join(output_path, f"process_cluster_{process_number}")
    os.makedirs(process_output_path, exist_ok=True)

    # Create the first cluster file inside the process_cluster
    cluster_number = 0
    batch_size = num_threads
    # batch_size = 1
    clusters_batches = []
    current_batch = {}

    for path, axiom in tqdm(zip(paths_0, axioms_0)):
        cluster_path = os.path.join(process_output_path, path.split("/")[-1])
        await copy_files(old_path=path, new_path=cluster_path)
        cluster_number += 1

        current_batch[cluster_path] = axiom 

        if len(current_batch) >= batch_size:
            clusters_batches.append(current_batch)
            current_batch = {}  

    # Append last batch also
    if current_batch:
        clusters_batches.append(current_batch)
        current_batch = {}     
    
    # Undefined files
    parent_path = os.path.dirname(output_path)
    undefined_path = os.path.join(parent_path, "undefined")

    # For showing only one process output
    show_progress = process_number == 0

    if show_progress:
        print(f"Start with {line_count_start} clusters.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as thread_executor:
        for indx, conj_line in tqdm(enumerate(conjs_1), disable=not show_progress):
            start_time = time.time()
            
            belongs_to_a_cluster = False
            contains_undefined_results = False

            vampire_calls = 0

            for batch in clusters_batches:
                cluster_path_batch = list(batch.keys())
                cluster_tptp_batch = list(batch.values())
        
                equivalence_result, cluster_path = check_equivalence_batch(cluster_path_batch, cluster_tptp_batch, conj_line, thread_executor, prover_path, timeout, show_progress, indx)
                vampire_calls += len(cluster_path_batch)
                
                if equivalence_result == 1:
                    belongs_to_a_cluster = True
                    # merge .txt and .tptp clusters
                    await merge_files(cluster_path, paths_1[indx])
                    break

                if equivalence_result == -1:
                    contains_undefined_results = True

            if not belongs_to_a_cluster and contains_undefined_results:
                # Write path to a undefined results file
                print("write to undefined")
                await write_to_files(undefined_path, paths_1[indx], paths_1[indx], mode="a")

            elif not belongs_to_a_cluster:
                # other cluster
                # print("Left as another cluster")
                cluster_number += 1
                new_cluster_path = os.path.join(process_output_path, f"cluster_{cluster_number}")
                await copy_files(old_path=paths_1[indx], new_path=new_cluster_path)
            
            if show_progress:
                print(f"Running line {indx} / {line_count_end}, Duration: {(time.time() - start_time) / 60:.1f} min with Vampire calls: {vampire_calls}, {cluster_number} clusters, Processed: {(time.time() - t) / 3600:.1f} hours.")
     

def main():
    parser = argparse.ArgumentParser(
        description="Iterate over a .tptp file in a folder, run pairwise Vampire checks in parallel with timeouts, and store clusters."
    )
    parser.add_argument("--input_path", required=True,
                        help="Path to a tptp file containing.")
    parser.add_argument("--output_path", required=True,
                        help="Path to a folder for storing clusters.")
    parser.add_argument("--prover_path", default="/auto/home/knarik/Molecular_Generation_with_GDB13/src/data/ACE_data/vampire",
                        help="Path to the Vampire/Eprover executable.")
    parser.add_argument("--chunk_size", type=int,
                        help="Chunk size is the size of the lines to process.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel processes for chunking (default=1).")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of parallel threads(default=1).")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout in seconds for each Vampire call (default=10).")
    # Parsing args
    args = parser.parse_args()

    log_args(args)

    input_path = args.input_path
    output_path = args.output_path
    prover_path = args.prover_path
    chunk_size = args.chunk_size
    num_workers = args.num_workers
    num_threads = args.num_threads
    timeout = args.timeout

    # Ensure output folder exists
    os.makedirs(output_path, exist_ok=True)

    # Track time for one file
    start_time = time.time()

    if os.path.isdir(input_path):
        print("Reading lines from clusters ...")

        process_clusters_paths = sorted(glob.glob(f"{input_path}/process_cluster_*"))
        process_clusters = []

        for path in tqdm(process_clusters_paths):
            clusters_tptp = sorted(glob.glob(f"{path}/*.tptp"))
            clusters_file_paths = [p.replace(".tptp", "") for p in clusters_tptp]

            axioms, conjs = load_tptp_from_clusters(clusters_tptp)
            process_clusters.append((axioms, conjs, clusters_file_paths))

        process_clusters_len = len(process_clusters)
    else:
        print("Reading lines from files ...")

        # Load tptp and txt files
        axioms, conjs = load_tptp(input_path)
        txt_lines = load_txt(input_path)
        axioms_len = len(axioms)
        print(f"There are {len(txt_lines)} lines.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as proecess_executor:
        futures = []

        if os.path.isdir(input_path):
            assert chunk_size == 2, "Chunk size should be equal to 2" 
            process_number = 0

            for start in range(0, process_clusters_len, chunk_size):

                # In case process_clusters_len is not divisable to chunk_size
                end = min(start + chunk_size, process_clusters_len)

                # Create chunks
                if end == process_clusters_len:
                    # Just copy the last cluster to a new place
                    process_output_path = os.path.join(output_path, f"process_cluster_{process_number}")
                    process_cluster_last_path = process_clusters[start: end][0][2][0]
                    process_cluster_last_path = os.path.dirname(process_cluster_last_path)

                    shutil.copytree(process_cluster_last_path, process_output_path)
                else:     
                    process_cluster_pair = process_clusters[start: end]

                    # Start processes
                    futures.append(
                        proecess_executor.submit(
                            process_cluster_chunk, 
                            process_number, 
                            process_cluster_pair[0], 
                            process_cluster_pair[1],
                            prover_path, 
                            num_threads,  
                            output_path, 
                            timeout
                        )
                    )
                    process_number += 1
        else:
            for start in range(0, axioms_len, chunk_size):

                # In case axioms_len is not divisable to chunk_size
                end = min(start + chunk_size, axioms_len)

                # Create chunks
                axioms_N = axioms[start: end]
                conjs_N = conjs[start: end]
                txt_N = txt_lines[start: end]

                # Start processes
                futures.append(
                    proecess_executor.submit(
                        process_chunk, 
                        start, 
                        axioms_N, 
                        conjs_N, 
                        txt_N, 
                        prover_path, 
                        num_threads, 
                        input_path, 
                        output_path, 
                        timeout 
                    )
                )

        # Get completed processes' results
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    print(f"Done. Processed all .tptp files. Time {(time.time() - start_time) / 3600} hours")


if __name__ == "__main__":
    main()