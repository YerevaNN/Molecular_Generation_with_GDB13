from tqdm import tqdm

subset = "equal_dist"
path = f"/auto/home/knarik/Molecular_Generation_with_GDB13/src/ablations/generations/generations/sf/generations_10M/unigram/rand_{subset}_unigram.csv"

with open(path, "r") as f_r:
    with open(path+"without.csv", "w") as f_w:
        for line_str in tqdm(f_r):
            new_line = line_str[:-5]
            
            f_w.write(new_line + "\n")