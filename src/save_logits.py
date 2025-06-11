import os
import time
import random
from tqdm import tqdm
from typing import Any

import numpy as np

import torch
import torch.nn as nn

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


@torch.no_grad()
def write_probs_csv(
    output_path: str,
    model: OPTForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    data_loader: Any
) -> None:
    
    print("Calculating logits ...")
    for i, inputs in tqdm(enumerate(data_loader)):
        targets = inputs[0][..., 1:] 
        inputs = inputs[0][..., :-1]
        batch_size = inputs.shape[0]
          
        if i == 0 and not os.path.exists(output_path):
            # created once
            t = time.time()
            max_seq_len = inputs.shape[1] 
            total_samples = 4
            # total_samples = len(data_loader.dataset)
            vocab_size = len(tokenizer.get_vocab())

            logits_memmap = np.lib.format.open_memmap(
                output_path, mode="w+", dtype=np.float32, shape=(total_samples, max_seq_len, vocab_size)
            )

            print(f"Created a memory-mapped npy file {output_path} in {(time.time() - t)} sec.")

        print(f"--------------- batch {i} -----------------------")
        # shape: [batch_size, seq_length, vocab_size]
        logits = model(inputs).logits    
        logits_np = logits.detach().cpu().numpy()
        logits_memmap[i*batch_size : i*batch_size + batch_size] = logits_np

        logits_memmap.flush()

        pad_id = tokenizer.convert_tokens_to_ids("<pad>")
        logits = logits.view(-1, logits.size(-1))   # shape: [batch_size * seq_length, vocab_size]
        targets = targets.contiguous().view(-1)     # shape: [batch_size * seq_length]
        print(logits)
        print("sum =============================================")
        print(logits.sum())

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs)        # shape: [batch_size * seq_length, vocab_size]

        # Calculate probabilities 
        loss = nn.NLLLoss(ignore_index=pad_id, reduction='none')
        log_token_probs = -loss(log_probs, targets)
        log_seq_probs = log_token_probs.view(inputs.size(0), -1)
        print("log_token_probs")
        print(log_token_probs)
        log_seq_probs = log_seq_probs.sum(-1)
        print("log_seq_probs")
        print(log_seq_probs)

        seq_probs = torch.exp((log_seq_probs)).tolist()  
        print(seq_probs)  
        print(targets.view(batch_size, -1))

    print(f"Logits have been successfully saved to {output_path}.")


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
        "--output_path",
        type=str,
        default=None,
        help="Output file for the validation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size of the validation data.",
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    seed_everything(2)

    data_path = args.data_path
    tokenizer_path = args.tokenizer_path
    checkpoint = args.resume_from_checkpoint
    output_path = args.output_path
    batch_size = args.batch_size

    # Prepare the tokenizer
    tokenizer = get_tokenizer(tokenizer_path, 192)
        
    # Get the dataloader
    time1 = time.time()
    data_loader = to_dataloader(path_to_data=data_path, tokenizer=tokenizer, batch_size=batch_size)
    print("Time for data creating:", time.time()-time1)

    # Get the model
    time1 = time.time()
    model = OPTForCausalLM.from_pretrained(pretrained_model_name_or_path=checkpoint)
    print("Time for forward:", time.time()-time1)

    # Prepare the dataloader and the model
    time1 = time.time()
    accelerator = Accelerator()
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval() 
    print("Time for accelerator:", time.time()-time1)
    
    # Write probabilities in the csv
    time1 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_probs_csv(output_path=output_path, model=model, tokenizer=tokenizer, data_loader=data_loader) 
    print("Time for writing:", time.time()-time1)