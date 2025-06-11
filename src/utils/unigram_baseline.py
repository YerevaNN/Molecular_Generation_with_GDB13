import json
import torch
import functools
from tqdm import tqdm


def main():
    subset_name = "aspirin_0.4"
    train_path = f"/auto/home/knarik/Molecular_Generation_with_GDB13/src/data/data/data_bin_all_rand_{subset_name}_sf_1000K/train/00/train_all_rand_{subset_name}_sf_1000K.jsonl"
    print("Reading from path", train_path)

    # Loading train set
    train_data = []

    with open(train_path, "r") as f:
        for line_str in tqdm(f):
            line_obj = json.loads(line_str)
            line_str = line_obj["text"]
            train_data.append( line_str +  "</s>")

    train_text = "".join(train_data)    

    # Loading vocabulary
    vocab_path = "/auto/home/knarik/Molecular_Generation_with_GDB13/src/data/tokenizers/tokenizer_sf/tokenizer.json"

    with open(vocab_path) as f:
        vocab_dict = json.load(f)

    vocab = [item["content"] for item in vocab_dict["added_tokens"]]

    # Counting train tokens
    train_len = train_text.count("[") + train_text.count("<")
    print("Train len", train_len)

    # Counting probabilities of each vocabulary token
    vocab_prob = []
    freq_dict = {}

    for token in tqdm(vocab):
        freq = train_text.count(token)
        freq_dict[token] = freq
        token_prob = freq / train_len

        vocab_prob.append(token_prob)

    print(freq_dict)
    exit()    

    # Sanity check
    prob_sum = 0
    for t, p in zip(vocab, vocab_prob):
        if p != 0:
            print(t,p)
            prob_sum += p    

    print("Sum of probabilities", prob_sum)

    # Generation process
    g = torch.Generator().manual_seed(678)
    generations = []
    max_seq_len = 64
    gen_len = 12_000_000

    for i in tqdm(range(gen_len)):
        gen_sample = []

        for i in range(max_seq_len):
            # Random sampling with replacement
            gen_token = torch.multinomial(torch.tensor(vocab_prob), num_samples=1, replacement=True, generator=g).item()
            
            if gen_token == 2:
                break

            gen_sample.append(vocab[gen_token])

        if vocab[2] in gen_sample and len(gen_sample) > 1:
            generations.append("".join(gen_sample)) 

    csv_file = open(f"rand_{subset_name}_unigram.csv", "wt+")
    write_func = functools.partial(csv_file.write)  
    write_func("\n".join(generations) + "\n")   
    
    csv_file.close()
    print(f"Saved in rand_{subset_name}_unigram.csv.")   


if __name__=="__main__":
    main()

    