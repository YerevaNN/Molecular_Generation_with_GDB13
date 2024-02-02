import os
import sys
import math
import csv
import json
import argparse
import functools
import parmap
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader
from accelerate import Accelerator
from tokenizers import Tokenizer
from datasets import load_dataset
from transformers import OPTForCausalLM, PreTrainedTokenizerFast


def read_file(path: str) -> list:
    with open(path, "r") as f:
        print(f"Reading file {path} ...")
        data = []

        for line_str in f:
            line_obj = json.loads(line_str)
            data.append(line_obj["text"])
    return data


def generate_fn(text, tokenizer, model):
        ids = tokenizer.encode(text, add_special_tokens=False)

        # Add <s> token and remove the [Canon]/[Rand] tokens
        input_prompt = torch.tensor([[0] + ids[1:]])
        
        # Generate shape(bs, seq_len)
        generate_ids = model.generate(input_prompt, max_length=64, do_sample=True)

        # Convert to list
        generate_ids = generate_ids.tolist()[0]
        output = tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return {
                "seq": text,
                "output": output
            }


def main():
    args = parse_args()

    # Convert to Transformer's tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_name)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.pad_token = "<pad>"

    if len(tokenizer.get_vocab()) != 192:
        i = len(tokenizer.get_vocab())

        while i < 192:
            symbol = "madeupword{:03d}".format(i)
        
            tokenizer.add_tokens(symbol)
            i += 1

    # Model
    model = OPTForCausalLM.from_pretrained(pretrained_model_name_or_path=args.resume_from_checkpoint)
    print(model)

    # if os.path.exists(args.output_dir):
    #     print(f"{args.output_dir} already exists")
    #     sys.exit(1)

    data = read_file(args.prompt_file)

    proceed_data = parmap.map(generate_fn, data, tokenizer, model, pm_processes=args.num_workers, pm_pbar=True)


    with open(args.output_dir, "w") as f:
        json.dump(proceed_data, f)
        print(f"Saved in {args.output_dir}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Pretrained model directory path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output file directory for the generations.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="",
        help="The prompt after <s> token.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size of the prompt.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel processes for parmap.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()