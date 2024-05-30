import json
import time
import argparse
from tqdm import tqdm
from chem_utils import get_canonical_form


def process_fn(line_str):
    try:
        line_obj = json.loads(line_str)

        # Transform to canon
        proceed_canon_str = get_canonical_form(line_obj["text"])

        if proceed_canon_str:
            # Write
            new_line = {"text": proceed_canon_str}

            return new_line
    except Exception as e:
        print(e)
        print("Line string", line_str)

    return None             


def read_lines_from_file(start_line, num_lines, input_file_name, output_file_name):
    start_time = time.time()

    with open(input_file_name, 'r') as file:
        # Skip to the start_line
        for _ in range(start_line - 1):
            next(file)

        print(f"Command Line Arguments: Start Line: {start_line}, Number of Lines: {num_lines}, File Name: {input_file_name}. Time taken to skip to start line: {time.time() - start_time:.2f} seconds")
        
        with open(output_file_name, "w") as file_2:
            for _ in range(num_lines):
                line = next(file, None)

                if line is None: 
                    break
    
                proceed_line = process_fn(line)  
    
                if proceed_line:
                    json.dump(proceed_line, file_2)
                    file_2.write("\n")
     

def main():
    parser = argparse.ArgumentParser(description='N/A')

    parser.add_argument('--start', type=int, help='Start line in the file.')
    parser.add_argument('--increment', type=int, help='Chunk size.')
    parser.add_argument('--input_file', type=str, help='Path to the input file')
    parser.add_argument('--output_file', type=str, help='Path to the output file')

    args = parser.parse_args()
    
    start = args.start
    increment = args.increment
    input_file_path = args.input_file
    output_file_path = args.output_file + str(start) + ".jsonl"

    print("Processing file: ", input_file_path)
    start_time = time.time()
    read_lines_from_file(start, increment, input_file_path, output_file_path)

    print("Runtime:", time.time() - start_time)


if __name__ == "__main__":
    main()