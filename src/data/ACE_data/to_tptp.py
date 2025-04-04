import time
import argparse
import subprocess
import concurrent.futures
from tqdm import tqdm
from pathlib import Path


def run_ape_tptp(sentence, ape_path="APE/ape.exe"):
    """
    Calls APE in TPTP mode for the given sentence.
    Returns a single-line string of the TPTP output
    (with newlines removed).
    """
    sentence = sentence.strip()
    if not sentence:
        return ""

    cmd = [ape_path, "-text", sentence, "-solo", "tptp"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error parsing sentence: {sentence}\n{e}")
        return "", ""
    except Exception as e:
        print(e)
        return "", ""

    # Combine all TPTP lines into one
    tptp_multiline = result.stdout
    tptp_single_line = " ".join(tptp_multiline.split())
    return tptp_single_line, sentence


def main():
    parser = argparse.ArgumentParser(
        description="Convert ACE sentences to TPTP (one line per sentence) using APE."
    )
    parser.add_argument("--input", required=True, help="Input file: one ACE sentence per line.")
    parser.add_argument("--output", required=True, help="Output file for TPTP lines.")
    parser.add_argument("--ape", default="APE/ape.exe",
                        help="Path to APE executable (default: APE/ape.exe).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: CPU count).")

    args = parser.parse_args()
    start_time = time.time()
    chunk_size = 5_000_000  # Save every 5 million lines
    count_error_lines = 0
    
    # Read all lines
    with open(args.input, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in tqdm(f)]
    
    total_lines = len(lines)
    num_chunks = (total_lines + chunk_size - 1) // chunk_size
    print("Starting processes")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for chunk_idx in tqdm(range(num_chunks)):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, total_lines)
            
            futures_map = {}
            print(f"Sumbitting sentences for chunk {chunk_idx} ...")

            for idx in tqdm(range(start, end)):
                sentence = lines[idx]
                fut = executor.submit(run_ape_tptp, sentence, args.ape)
                futures_map[fut] = idx
            
            # Collect results
            results_tptp = [None] * (end - start)
            results_txt = [None] * (end - start)
            for fut in tqdm(concurrent.futures.as_completed(futures_map), total=len(futures_map), desc=f"Processing chunk {chunk_idx+1}/{num_chunks}"):
                idx = futures_map[fut]
                results_tptp[idx - start], results_txt[idx - start] = fut.result()
            
            # Write results to file immediately
            mode = "w" if chunk_idx == 0 else "a"  # Overwrite first chunk, append others
            output_txt_path = str(Path(args.output).with_suffix("")) + ".txt"

            with open(args.output, mode, encoding="utf-8") as out_tptp:
                with open(output_txt_path, mode, encoding="utf-8") as out_txt:
                    for line_tptp, line_txt in zip(results_tptp, results_txt):
                        if "xml" in line_tptp:
                            count_error_lines += 1
                            
                            print("Not parsable sentence.")
                            print(line_txt)
                            continue

                        if line_tptp:
                            out_tptp.write(line_tptp + "\n")
                            out_txt.write(line_txt + "\n")

    
    print(f"Done. {total_lines} lines processed in {num_chunks} chunks. Number of Error lines: {count_error_lines}. Time: {time.time() - start_time} sec.")


if __name__ == "__main__":
    main()