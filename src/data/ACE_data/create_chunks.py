import os

def chunk_and_save_tptp(txt_path, tptp_path, output_dir, chunk_size=300_000):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    chunk_idx = 0
    current_chunk_tptp = []
    current_chunk_txt = []

    with open(txt_path, "r") as txt_file, open(tptp_path, "r") as tptp_file:
        for txt_line, tptp_line in zip(txt_file, tptp_file):
            current_chunk_tptp.append(tptp_line)
            current_chunk_txt.append(txt_line)

            # When the chunk reaches the specified size, save it and reset
            if len(current_chunk_tptp) >= chunk_size:
                output_file = os.path.join(output_dir, f"big-card-customer-inserts-small-uses_chunk_{chunk_idx}.tptp")
                with open(output_file, "w") as out_file:
                    out_file.writelines(current_chunk_tptp)

                output_file_txt = os.path.join(output_dir, f"big-card-customer-inserts-small-uses_chunk_{chunk_idx}.txt")
                with open(output_file_txt, "w") as out_file:
                    out_file.writelines(current_chunk_txt)    
                
                # Reset for the next chunk
                current_chunk_tptp = []
                current_chunk_txt = []
                chunk_idx += 1

        # Save any remaining lines as the last chunk
        if current_chunk_tptp:
            output_file = os.path.join(output_dir, f"big-card-customer-inserts-small-uses_chunk_{chunk_idx}.tptp")
            with open(output_file, "w") as out_file:
                out_file.writelines(current_chunk_tptp)

            output_file_txt = os.path.join(output_dir, f"big-card-customer-inserts-small-uses_chunk_{chunk_idx}.txt")
            with open(output_file_txt, "w") as out_file:
                out_file.writelines(current_chunk_txt)        

    print(f"Chunking complete. Created {chunk_idx + 1} files in {output_dir}")

# File paths
txt_path = "/nfs/h100/raid/chem/cf_17/big-card-customer-inserts-small-uses.txt"
tptp_path = "/nfs/h100/raid/chem/cf_17/big-card-customer-inserts-small-uses.tptp"
output_dir = "/nfs/h100/raid/chem/cf_17/"

# Run the chunking process
chunk_and_save_tptp(txt_path, tptp_path, output_dir)
