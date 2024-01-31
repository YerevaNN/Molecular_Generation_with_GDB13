import os
import sys
import math
import argparse
import functools
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader
from accelerate import Accelerator
from tokenizers import Tokenizer
from transformers import OPTForCausalLM, PreTrainedTokenizerFast


def main():
    args = parse_args()

    accelerator = Accelerator()

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

    # Prompt, includes BOS token
    start_prompt = [0]

    if args.prompt_token != "_":
        id = tokenizer.convert_tokens_to_ids(args.prompt_token)
        start_prompt.append(id)

    prompt = torch.tensor(start_prompt, dtype=int)
    print("The prompt is", prompt.data)
    prompt = prompt.repeat(args.batch_size, 1)

    dataset = TensorDataset(prompt)
    data_loader = DataLoader(dataset, args.batch_size, shuffle=False)

    # Model 
    model = OPTForCausalLM.from_pretrained(pretrained_model_name_or_path=args.resume_from_checkpoint)
    print(model)

    model, data_loader = accelerator.prepare(model, data_loader)

    # Output file 
    if os.path.exists(args.output_dir):
        print(f"{args.output_dir} already exists")
        sys.exit(1)

    csv_file = open(args.output_dir, "wt+")
    write_func = functools.partial(csv_file.write)

    data_iter = iter(data_loader)
    prompt_data = next(data_iter)[0]

    for i in tqdm(range(math.ceil(args.gen_len / args.batch_size))):

        # Generate
        generate_ids = model.generate(prompt_data, max_length=64, do_sample=True)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Write in a file
        write_func("\n".join(outputs) + "\n")

    csv_file.close()
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
        "--prompt_token",
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
        "--gen_len",
        type=int,
        default=1,
        help="The number of generations.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()