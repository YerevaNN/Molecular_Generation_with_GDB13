import os
import re
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

import numpy as np

import cProfile
import pstats
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

        logging.info(f"Count of axioms is {len(axioms)}. Time {time.time() - time_start:.3f} sec.")   

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

    logging.info(f"Count of axioms is {len(axioms)}. Time {time.time() - time_start:.3f} sec.")   

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

        logging.info(f"Count of lines is {len(txt_lines)}. Time {time.time() - time_start:.3f} sec.")   

    return txt_lines 


def load_txt_from_clusters(input_paths):
    time_start = time.time()
    txt_lines = []

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()

            if line:
                txt_lines.append(line)

    logging.info(f"Count of lines is {len(txt_lines)}. Time {time.time() - time_start:.3f} sec.")   

    return txt_lines 
        
    
def run_vampire(axiom_line, conj_line, vampire_path, timeout=10, attempt=0):
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
        logging.info(f"Unexpected Vampire output: `{output}` at attempt {attempt}")
        if attempt == 3:
            return -1
        return run_vampire(axiom_line, conj_line, vampire_path, timeout=50, attempt=attempt + 1)


def worker_fn(task, prover_path, timeout):
    """Top-level function for parallel calls."""
    indx, ax_line, conj_line = task

    if "vampire" in prover_path:
        res = run_vampire(ax_line, conj_line, prover_path, timeout=timeout)  
    else:
        raise NameError("There is no such a prover")    
    return (indx,  res)


def write_line_to_file(
    path: str,
    line: str,
    mode: str = "w",
    extension: str = "txt",
    retries: int = 0,
    max_retries: int = 3
) -> None:
    try:
        with open(f"{path}.{extension}", mode) as f:
            f.write(line + "\n")

        # if mode == "w":
        #     logging.info(f"Data saved to {path}")

    except Exception as e:
        if retries >= max_retries:
            raise OSError(f"Failed after {max_retries} retries: {e}")

        return write_line_to_file(path, line, mode, extension, retries + 1, max_retries)


def write_to_files(cluster_path, txt_line, tptp_line, mode="w"):
    # write to txt file
    write_line_to_file(cluster_path, txt_line, mode=mode, extension="txt"), 

    # write to tptp file
    if "conjecture" in tptp_line:
        tptp_line = tptp_line.replace(", conjecture,", ", axiom,")
        
    write_line_to_file(cluster_path, tptp_line, mode=mode, extension="tptp")  


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

    # delete files
    # await asyncio.gather(
    #     async_remove_file(f"{merge_file_path}.txt"),
    #     async_remove_file(f"{merge_file_path}.tptp"),
    # )


async def copy_files(new_path: str, old_path: str):
    """Asynchronously copy files and then remove the originals."""
    # Copy files
    await asyncio.gather(
        async_copy_file(f"{old_path}.txt", f"{new_path}.txt"),
        async_copy_file(f"{old_path}.tptp", f"{new_path}.tptp"),
    )

    # Remove old files
    # await asyncio.gather(
    #     async_remove_file(f"{old_path}.txt"),
    #     async_remove_file(f"{old_path}.tptp"),
    # )


async def async_remove_file(path: str):
    """Asynchronously remove a file."""
    try:
        await asyncio.to_thread(os.remove, path)
        # print(f"Deleted: {path}")
    except Exception as e:
        print(f"Error deleting {path}: {e}")


async def async_copy_file(src: str, dst: str):
    """Asynchronously copy a file using a thread to avoid blocking the event loop."""
    await asyncio.to_thread(shutil.copy, src, dst)


def check_equivalence(axiom_line, conj_line, executor, prover_path, timeout):
    equivalence_result = [-1, -1]

    current_pair = [[0, axiom_line, conj_line], [1, conj_line, axiom_line]]   

    futures = [
        executor.submit(worker_fn, t, prover_path, timeout)
        for t in current_pair
    ]

    for fut in concurrent.futures.as_completed(futures):
        indx, res_val = fut.result()
        equivalence_result[indx] = res_val

        if res_val == -1:
            logging.error(f"Unexpected value for axiom ({axiom_line})")

    return equivalence_result   


def process_chunk(start, axioms_N, conjs_N, txt_N, prover_path, num_threads, input_path, output_path, timeout):
    t = time.time()
    line_count = len(axioms_N)
    
    # Each chunk creates a process_cluster for its own
    process_output_path = os.path.join(output_path, f"process_cluster_{start}")
    os.makedirs(process_output_path, exist_ok=True)
    
    # Create the first cluster file inside the process_cluster
    cluster_number = 0
    cluster_path = os.path.join(process_output_path, f"cluster_{cluster_number}")
    clusters = {cluster_path: axioms_N[0]}
    write_to_files(cluster_path, txt_N[0], axioms_N[0], mode="w") 

    undefined_path = os.path.join(output_path, "undefined")

    if not os.path.exists(os.path.join(output_path, "undefined.txt")):
        # Create a file for undefined (-1) equivalence results also
        write_to_files(undefined_path, "", "", mode="w") 

    # For showing only one process output
    show_progress = start == 0

    # if show_progress:  
    #     profiler = cProfile.Profile()
        # profiler.enable()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as thread_executor:
        for indx, conj_line in tqdm(enumerate(conjs_N), disable=not show_progress):
            start_time = time.time()
            
            if indx == 0:
                continue
            
            belongs_to_a_cluster = False
            contains_undefined_results = False

            vampire_calls = 0

            for cluster_path, cluster_tptp in clusters.items():    
                equivalence_result = check_equivalence(cluster_tptp, conj_line, thread_executor, prover_path, timeout)
                vampire_calls += 1
                
                if equivalence_result[0] == equivalence_result[1] == 1:
                    belongs_to_a_cluster = True
                    # write to .txt and .tptp files
                    write_to_files(cluster_path, txt_N[indx], conj_line, mode="a")
                    break

                if equivalence_result[0] == -1 or equivalence_result[1] == -1:
                    contains_undefined_results = True

            if not belongs_to_a_cluster and contains_undefined_results:
                # Write to a undefined results file
                write_to_files(undefined_path, txt_N[indx], conj_line, mode="a")
            elif not belongs_to_a_cluster:
                # Create new cluster
                cluster_number += 1
                new_cluster_path = os.path.join(process_output_path, f"cluster_{cluster_number}")
                axiom_line = conj_line.replace(", conjecture,", ", axiom,")
                clusters[new_cluster_path] = axiom_line
                write_to_files(cluster_path, txt_N[indx], axiom_line, mode="w")
            
            if show_progress:
                print(f"Running line {indx} / {line_count}, Duration: {(time.time() - start_time) / 60:.1f} min with {cluster_number} clusters, Vampire calls: {vampire_calls}, Processed: {(time.time() - t) / 3600:.1f} hours.")

    # if show_progress:  
    #     profiler.disable()
    #     stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    #     stats.print_stats(20)


def process_cluster_chunk(start, axioms_N, conjs_N, txt_N, clusters_paths_N, prover_path, num_threads, input_path, output_path, timeout):
    t = time.time()
    line_count = len(axioms_N)

    # Each chunk creates a process_cluster for its own
    process_output_path = os.path.join(output_path, f"process_cluster_{start}")
    os.makedirs(process_output_path, exist_ok=True)

    # Create the first cluster file inside the process_cluster
    cluster_number = 0
    cluster_path = os.path.join(process_output_path, f"cluster_{cluster_number}")

    clusters = {cluster_path: axioms_N[0]}
    # asyncio.run(copy_and_remove_files(old_path=clusters_paths_N[0], new_path=cluster_path))
    asyncio.run(copy_files(old_path=clusters_paths_N[0], new_path=cluster_path))

    # Undefined files
    parent_path = os.path.dirname(output_path)
    undefined_path = os.path.join(parent_path, "undefined")

    # For showing only one process output
    show_progress = start == 0

    # if show_progress:  
    #     profiler = cProfile.Profile()
        # profiler.enable()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as thread_executor:
        for indx, conj_line in tqdm(enumerate(conjs_N), disable=not show_progress):
            start_time = time.time()
            
            if indx == 0:
                continue
            
            belongs_to_a_cluster = False
            contains_undefined_results = False

            vampire_calls = 0

            for cluster_path, cluster_tptp in clusters.items():    
                # skip if they are within the same process_cluster
                cluster_path_match = re.search(r"process_cluster_\d+", cluster_path)
                merge_file_path_match = re.search(r"process_cluster_\d+", clusters_paths_N[indx])

                if cluster_path_match.group() == merge_file_path_match.group():
                    # print("skip comparison", cluster_path_match.group(), merge_file_path_match.group())
                    continue

                equivalence_result = check_equivalence(cluster_tptp, conj_line, thread_executor, prover_path, timeout)
                vampire_calls += 1
                
                if equivalence_result[0] == equivalence_result[1] == 1:
                    belongs_to_a_cluster = True
                    # merge .txt and .tptp clusters
                    asyncio.run(merge_files(cluster_path, clusters_paths_N[indx]))
                    break

                if equivalence_result[0] == -1 or equivalence_result[1] == -1:
                    contains_undefined_results = True

            if not belongs_to_a_cluster and contains_undefined_results:
                # Write path to a undefined results file
                print("write to undefined")
                write_to_files(undefined_path, clusters_paths_N[indx], clusters_paths_N[indx], mode="a")

            elif not belongs_to_a_cluster:
                # other cluster
                # print("Left as another cluster")
                cluster_number += 1
                new_cluster_path = os.path.join(process_output_path, f"cluster_{cluster_number}")
                axiom_line = conj_line.replace(", conjecture,", ", axiom,")

                clusters[new_cluster_path] = axiom_line
                asyncio.run(copy_files(old_path=clusters_paths_N[indx], new_path=new_cluster_path))
            
            if show_progress:
                print(f"Running line {indx} / {line_count}, Duration: {(time.time() - start_time) / 60:.1f} min with {cluster_number} clusters, Vampire calls: {vampire_calls}, Processed: {(time.time() - t) / 3600:.1f} hours.")

    # if show_progress:  
    #     profiler.disable()
    #     stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    #     stats.print_stats(20)
     

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
    parser.add_argument("--timeout", type=int, default=10,
                        help="Timeout in seconds for each Vampire call (default=50).")
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
    clusters_paths = []

    if os.path.isdir(input_path):
        print("Reading lines from clusters ...")
        # There are already clusters
        clusters_tptp_paths = sorted(glob.glob(f"{input_path}/process_cluster_*/*.tptp"))
        clusters_txt_paths = sorted(glob.glob(f"{input_path}/process_cluster_*/*.txt"))
        clusters_paths = [p.replace(".tptp", "") for p in clusters_tptp_paths]

        # Generating permutation order
        perm = np.random.permutation(len(clusters_paths))

        # Shuffle order
        clusters_tptp_paths = np.array(clusters_tptp_paths)[perm].tolist()
        clusters_txt_paths = np.array(clusters_txt_paths)[perm].tolist()
        clusters_paths = np.array(clusters_paths)[perm].tolist()

        # Load data
        axioms, conjs = load_tptp_from_clusters(clusters_tptp_paths)
        txt_lines = load_txt_from_clusters(clusters_txt_paths)
        axioms_len = len(axioms)
        print(f"There are {len(txt_lines)} clusters.")
    else:
        print("Reading lines from files ...")
        # Load tptp and txt files
        axioms, conjs = load_tptp(input_path)
        txt_lines = load_txt(input_path)
        axioms_len = len(axioms)
        print(f"There are {len(txt_lines)} lines.")

    # Start the CPU usage monitor thread
    # cpu_monitor = CPUUsageMonitor(interval=1)
    # cpu_monitor.start()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as proecess_executor:
        futures = []

        for start in range(0, axioms_len, chunk_size):
        # for start in range(0, 90, chunk_size):
            # in case axioms_len is not divisable to chunk_size
            end = min(start + chunk_size, axioms_len)

            # create chunks
            axioms_N = axioms[start: end]
            conjs_N = conjs[start: end]
            txt_N = txt_lines[start: end]

            if clusters_paths:
                clusters_paths_N = clusters_paths[start: end]

                # start processes
                futures.append(
                    proecess_executor.submit(
                        process_cluster_chunk, 
                        start, 
                        axioms_N, 
                        conjs_N, 
                        txt_N, 
                        clusters_paths_N,
                        prover_path, 
                        num_threads, 
                        input_path, 
                        output_path, 
                        timeout
                    )
                )
            else:
                # start processes
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

        # get completed processes' results
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    # Stop the CPU monitor and get the average CPU usage
    # cpu_monitor.stop()
    # cpu_monitor.join()
    # avg_cpu = cpu_monitor.get_average_usage()

    print(f"Done. Processed all .tptp files. Time {(time.time() - start_time) / 3600} hours")
    # print(f"Average CPU usage across all cores: {avg_cpu:.2f}%")


if __name__ == "__main__":
    main()