import os
import json
import selfies as sf
from tqdm import tqdm
from rdkit import Chem


if __name__ == "__main__":
    subset_name = "sas_3"

    for split in ["train"]:
        folder_path = f"../data/data/data_bin_{subset_name}_sm_1000K/{split}/00/"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(f"../data/data/data_bin_all_canon_{subset_name}_sf_1000K/{split}/00/{split}_all_canon_{subset_name}_sf_1000K.jsonl", "r") as file_1:
            with open(f"../data/data/data_bin_all_canon_{subset_name}_sm_1000K/{split}/00/{split}_all_canon_{subset_name}_sm_1000K.jsonl", "w") as file_2:
                for line_str in tqdm(file_1):
                    line_obj = json.loads(line_str)
                    # Transform to smiles
                    sm_str = sf.decoder(line_obj["text"])
                    sm_str_canon = Chem.MolToSmiles(Chem.MolFromSmiles(sm_str), canonical=True, kekuleSmiles=False)
                    # Write
                    new_line = {"text": sm_str_canon}
                    json.dump(new_line, file_2)
                    file_2.write("\n")

        # with open(f"../data/data/data_bin_all_canon_{subset_name}_sf_1000K/{split}/00/valid_all_rand_{subset_name}_sf_0.5K_rand_all_versions.jsonl", "r") as file_1:
        #     with open(f"../data/data/data_bin_all_canon_{subset_name}_sm_1000K/{split}/00/valid_all_rand_{subset_name}_sm_0.5K_rand_all_versions.jsonl", "w") as file_2:
        #         for line_str in tqdm(file_1):
        #             line_obj = json.loads(line_str)
        #             # Transform to smiles
        #             sm_str = sf.decoder(line_obj["text"])
        #             sm_str_rand = Chem.MolToSmiles(Chem.MolFromSmiles(sm_str), canonical=False, kekuleSmiles=False)
        #             # Write
        #             new_line = {"text": sm_str_rand}
        #             json.dump(new_line, file_2)
        #             file_2.write("\n")            