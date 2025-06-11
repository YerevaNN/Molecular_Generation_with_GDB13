import os 
import sys
import csv
import time
import random
import numpy as np
from tqdm import tqdm
from typing import Any
from typing import Union, Optional

import torch
import torch.nn as nn
from torch import Tensor

import argparse
from accelerate import Accelerator
from transformers import OPTForCausalLM, PreTrainedTokenizerFast, LlamaForCausalLM

from utils.get_tokenizer import get_tokenizer
from utils.data_preprocessing import to_dataloader
from utils.sm_sf_processing import rand_to_canon_str


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_probs_csv(
    in_path: str,
    output_path: str,
    tokenizer: PreTrainedTokenizerFast,
    data_loader: Any,
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float],
    save_type: str,
    str_type: str,
) -> None:

    if os.path.exists(output_path):
        print(f"Error: {output_path} already exists.")
        sys.exit(1)

    pad_id = tokenizer.convert_tokens_to_ids("<pad>")

    # Open the .npy file as a memory-mapped array
    all_logits = np.lib.format.open_memmap(in_path, mode="r")

    with open(output_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Probability"])
        
        print("Calculating probabilities ...")

        for i, inputs in tqdm(enumerate(data_loader)):
            targets = inputs[0][..., 1:]

            batch_size = targets.shape[0]
            logits = torch.tensor(all_logits[i*batch_size : i*batch_size + batch_size]).to(targets.device)

            # Decode target sequence
            # Convert to canonical string if specified
            target_seq = []
            for sequence_ids in targets:
                decoded_seq = tokenizer.decode(sequence_ids, skip_special_tokens=True)
                if save_type == 'CANON':
                    decoded_seq = rand_to_canon_str(sequence=decoded_seq, str_type=str_type)
                target_seq.append(decoded_seq)

            logits = logits.view(-1, logits.size(-1))   # shape: [batch_size * seq_length, vocab_size]
            targets = targets.contiguous().view(-1)

            # Get modified (wrt sampling strategy) probabilities
            if temperature != 1.0:
                logits /= temperature

            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs)        # shape: [batch_size * seq_length, vocab_size]

            # Calculate probabilities 
            loss = nn.NLLLoss(ignore_index=pad_id, reduction='none')
            log_token_probs = -loss(log_probs, targets)
            log_seq_probs = log_token_probs.view(batch_size, -1)
            log_seq_probs = log_seq_probs.sum(-1)

            seq_probs = torch.exp((log_seq_probs)).tolist()   

            rows = [[str_repr, prob] for str_repr, prob in zip(target_seq, seq_probs)] 

            writer.writerows(rows)


    # Close the memory-mapped file
    del all_logits
    print(f"Probabilities have been successfully saved to {output_path}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a model wrt decoding strategy.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="The path of validation data"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Pretrained tokenizer path",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Pretrained model directory path.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="The parameter of top_k sampling. The default value is the length of vocabulary."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="The parameter of top_p sampling."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of sampling."
    )
    parser.add_argument(
        "--save_type",
        type=str,
        default=None,
        help="The type of SELFIES CANON/RAND"
    )
    parser.add_argument(
        "--str_type",
        type=str,
        default='selfies',
        help="The type of molecular representation"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output file for the logits.",
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default=None,
        help="Input file for the logits.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size of the logits data.",
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()
    seed_everything(2)

    accelerator = Accelerator()

    data_path = args.data_path
    tokenizer_path = args.tokenizer_path
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    save_type = args.save_type
    str_type = args.str_type
    in_path = args.in_path
    output_path = args.output_path
    batch_size = args.batch_size

    # Prepare the tokenizer
    tokenizer = get_tokenizer(tokenizer_path, 192)
        
    # Get the dataloader
    time1 = time.time()
    data_loader = to_dataloader(path_to_data=data_path, tokenizer=tokenizer, batch_size=batch_size)
    print("Time for data creating:", time.time()-time1)

    # Prepare the dataloader and the model
    time1 = time.time()
    data_loader = accelerator.prepare(data_loader)
    print("Time for accelerator:", time.time()-time1)
    
    # Write probabilities in the csv 
    time1 = time.time()
    write_probs_csv(in_path=in_path, output_path=output_path, tokenizer=tokenizer, data_loader=data_loader, top_k=top_k, top_p=top_p, temperature=temperature, save_type=save_type, str_type=str_type) 
    print(f"Overall Time: {(time.time()-time1) // 60} sec.")