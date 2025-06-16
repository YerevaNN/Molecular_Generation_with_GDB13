from tqdm import tqdm
import json

from transformers import AutoTokenizer



def read_jsonl(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = []
        for sample in tqdm(f):
            json_line = json.loads(sample)["text"]
            data.append(json_line)

    return data        


def main():
    path = "/auto/home/knarik/Molecular_Generation_with_GDB13/src/data/data/data_bin_python/train/train.jsonl"
    tokenizer_path = "meta-llama/Llama-3.2-1B"

    data = read_jsonl(path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)   

    vocab = set()
    for line in tqdm(data):
        vocab = vocab.union(tokenizer(line)["input_ids"])

    with open("/auto/home/knarik/Molecular_Generation_with_GDB13/src/ablations/generations/generations/code/train_vocab.txt", "w") as f:
        for el in vocab:
            f.write(f"{el}\n")


if __name__ == "__main__":
    main()