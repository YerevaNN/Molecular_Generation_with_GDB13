import json
import time
import argparse
import selfies as sf
from tqdm import tqdm
from chem_utils import get_canonical_form


def count_intesection_by_molecule(gen_file, train_file, repr="sf"):
    train_data = []
    gen_num = 1_000_000
    from_train_set = set()

    print("Processing generation file: ", gen_file)

    with open(train_file, 'r') as t_file:
        for line_str in tqdm(t_file):
            line_obj = json.loads(line_str)
            canon_form_train = line_obj["text"]
            # if canon_form_train != get_canonical_form(canon_form_train):
            #     print("!!!!!!!!!!!!!! not canonical")
            train_data.append(canon_form_train)
            
    print("Processing trainig file: ", train_file)

    with open(gen_file, 'r') as g_file:
        g_data = g_file.read().splitlines()

        for line_str in tqdm(g_data[:gen_num]):
            if repr == "sf":
            # Transform to smiles
                line_str = sf.decoder(line_str)

            # Transform to canon
            canon_form_gen = get_canonical_form(line_str)

            if repr == "sf":
                # Transform to selfies
                canon_form_gen = sf.encoder(canon_form_gen)    

            # Count generations that are in train subset
            if canon_form_gen in train_data:
                from_train_set.add(canon_form_gen)
     
    return len(from_train_set)
                       

def main():
    parser = argparse.ArgumentParser(description='N/A')
    parser.add_argument('--gen_file', type=str, help='Path to the generation file')
    parser.add_argument('--train_file', type=str, help='Path to the training file, which is CANONICAL')
    parser.add_argument('--repr', type=str, help='sm or sf')
    args = parser.parse_args()

    start_time = time.time()
    inter_count = count_intesection_by_molecule(args.gen_file, args.train_file, repr=args.repr)

    print("Runtime:", time.time() - start_time)
    print("Intersection count is", inter_count)  


if __name__ == "__main__":
    main()
