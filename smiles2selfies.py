import selfies as sf
import json
from tqdm import tqdm


if __name__ == "__main__":
    # substr = "sas_7"
    substr = "original"

    for split in ["train", "valid"]:
        with open(f"./data-bin/data-subsets/{split}_{substr}.jsonl", "r") as file_1:
            with open(f"./data-bin/data-subsets/{split}_{substr}_selfies.jsonl", "w") as file_2:
                for line_str in tqdm(file_1):
                    line_obj = json.loads(line_str)
                    # Transform to selfies
                    selfies_str = sf.encoder(line_obj["text"])
                    # Write
                    new_line = {"text": selfies_str}
                    json.dump(new_line, file_2)
                    file_2.write("\n")