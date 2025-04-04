#!/usr/bin/env python3

import argparse
import os
from tqdm import tqdm

VOCAB = ['customer', 'card', 'big', 'small', 'inserts', 'uses']

def main():
    parser = argparse.ArgumentParser(
        description="Group sentence+TPTP pairs by bag-of-words from a fixed vocabulary."
    )
    parser.add_argument("--text", required=True, help="Path to the text file (sentences).")
    parser.add_argument("--tptp", required=True, help="Path to the TPTP file (same # of lines).")
    parser.add_argument("--output", required=True, help="Output folder.")
    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output, exist_ok=True)

    # Count lines in the text file to display progress
    with open(args.text, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # We'll keep open file handles in a dictionary
    file_handles = {}

    # Process both files in parallel, line by line, using tqdm
    with open(args.text, "r", encoding="utf-8") as text_f, \
         open(args.tptp, "r", encoding="utf-8") as tptp_f, \
         tqdm(total=total_lines, desc="Processing lines") as pbar:

        for text_line, tptp_line in tqdm(zip(text_f, tptp_f)):
            sentence = text_line.strip()
            tptp = tptp_line.strip()

            # Extract words from the vocabulary
            words_in_sent = set()
            for w in sentence.lower().strip()[:-1].split():
                if w in VOCAB:
                    words_in_sent.add(w)

            # Skip if no relevant vocabulary
            if not words_in_sent:
                pbar.update(1)
                continue

            # Sort them so that the file name is deterministic (e.g. 'big-card-customer')
            bow_sorted = sorted(words_in_sent)
            bow_str = "-".join(bow_sorted)

            # Open or reuse file handles for bow_str
            if bow_str not in file_handles:
                txt_path = os.path.join(args.output, f"{bow_str}.txt")
                tptp_path = os.path.join(args.output, f"{bow_str}.tptp")

                txt_handle = open(txt_path, "a", encoding="utf-8")
                tptp_handle = open(tptp_path, "a", encoding="utf-8")
                file_handles[bow_str] = (txt_handle, tptp_handle)

            # Write to the relevant files
            txt_handle, tptp_handle = file_handles[bow_str]
            txt_handle.write(sentence + "\n")
            tptp_handle.write(tptp + "\n")

            # Update progress
            pbar.update(1)

    # Close all file handles
    for _, (txt_handle, tptp_handle) in file_handles.items():
        txt_handle.close()
        tptp_handle.close()

    print("Done. Grouped files have been written to:", args.output)


if __name__ == "__main__":
    main()
