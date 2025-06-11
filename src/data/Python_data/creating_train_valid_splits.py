import os
import json
import random
import argparse
from tqdm import tqdm


def read_code_samples(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines, assuming samples are separated by empty lines
    samples = [sample.strip() for sample in content.strip().split('\n\n') if sample.strip()]

    return samples


def write_jsonl(samples, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sample in tqdm(enumerate(samples)):
            json_line = json.dumps({"text": sample}, ensure_ascii=False)
            f.write(json_line + '\n')


def main(input_path, output_dir, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    train_size = 100_000

    samples = read_code_samples(input_path)

    random.seed(seed)
    random.shuffle(samples)

    train_samples = samples[:train_size]
    val_samples = samples[train_size: train_size+10_000]

    train_path = os.path.join(output_dir, 'train_1.jsonl')
    val_path = os.path.join(output_dir, 'val.jsonl')

    write_jsonl(train_samples, train_path)
    write_jsonl(val_samples, val_path)
    print(f"Wrote {len(train_samples)} training samples to {train_path}")
    print(f"Wrote {len(val_samples)} validation samples to {val_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input .txt file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save train/val .jsonl files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    args = parser.parse_args()

    main(args.input_path, args.output_dir, args.seed)