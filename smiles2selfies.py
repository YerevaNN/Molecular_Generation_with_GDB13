import os
import json
import selfies as sf
from tqdm import tqdm


if __name__ == "__main__":
    subset_name = "equal_dist"

    for split in ["train", "valid"]:
        folder_path = f"./data/data_bin_{subset_name}_sf_1000K/{split}/00/"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(f"./data/data_bin_{subset_name}_sm_1000K/{split}/00/{split}_{subset_name}_smiles.jsonl", "r") as file_1:
            with open(f"./data/data_bin_{subset_name}_sf_1000K/{split}/00/{split}_{subset_name}_smiles.jsonl", "w") as file_2:
                for line_str in tqdm(file_1):
                    line_obj = json.loads(line_str)
                    # Transform to selfies
                    selfies_str = sf.encoder(line_obj["text"])
                    # Write
                    new_line = {"text": selfies_str}
                    json.dump(new_line, file_2)
                    file_2.write("\n")