import os
import csv
import time
import argparse
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import OPTForCausalLM, LlamaForCausalLM, AutoTokenizer
from accelerate import Accelerator
from torch import manual_seed

from utils.get_tokenizer import get_tokenizer
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

BOS = 128000
EOS = 128001
PAD = 11111 # random token

def parse_args():
    parser = argparse.ArgumentParser(description="Generate strings with beam search.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Already perpared vocabulary.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Vocab size, tokenizer size with all added tokens.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Pretrained model directory path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size of the validation data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The path of output beams."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature."
    )
    parser.add_argument(
        "--gen_len",
        type=int,
        default=1_000_000,
        help="The length of generation."
    )
    parser.add_argument(
        "--iter_len",
        type=int,
        default=33,
        help="The length of iterations."
    )
    args = parser.parse_args()

    return args


def write_tokens_to_csv(output_path, hypothesis_tokens, hypothesis_probs, tokenizer):
    with open(output_path, 'w', newline='') as csvfile:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = csv.writer(csvfile)
        writer.writerow(['Hypothesis', 'NLL'])

        # Iterate over each hypothesis
        print(f"Writing in a file {output_path} ...")
        for tokens, probs in tqdm(zip(hypothesis_tokens, hypothesis_probs)):
            
            hypothesis_text = tokenizer.decode(tokens, skip_special_tokens=True)

            writer.writerow([hypothesis_text, probs])  


def to_data_loader(vocab, batch_size):
    inputs = []
    targets = []
    sequences = []

    for token_1 in vocab:
        if token_1 in [BOS, EOS, PAD]:
            continue

        for token_2 in vocab:
            if token_2 in [BOS, PAD]:
                continue

            inputs.append([BOS, token_1])
            targets.append([token_1, token_2])
            sequences.append([BOS, token_1, token_2])    

    dataset = TensorDataset(torch.tensor(inputs), torch.tensor(targets))
    data_loader = DataLoader(dataset, batch_size, shuffle=False)

    return data_loader, sequences


@torch.no_grad()
def beam_decoding(model, accelerator, batch_size, vocab, iter_len, gen_len, temp):
    time1 = time.time()
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    vocab_len = len(vocab)
    vocab = torch.tensor(vocab, dtype=torch.int64)

    # Get dataloader
    data_loader, sequences = to_data_loader(vocab, batch_size)

    # Give to the acclerator
    model, data_loader = accelerator.prepare(model, data_loader)

    log_probs = []

    for input, target in tqdm(data_loader):
        # shape: (batch_size, 2, vocab len)
        logits = model(input).logits

        # shape: (batch_size x 2, vocab len)
        flatten_logits = logits.view(-1, logits.size(-1))

        # shape: (batch_size x 2)
        targets = target.view(-1)

        # shape: (batch_size x 2)
        flat_log_probs = -loss(flatten_logits, targets)
        
        # shape: (batch_size, 2)
        flat_log_probs_sum = flat_log_probs.view(-1, 2).sum(dim=-1).tolist()

        log_probs.extend(flat_log_probs_sum)   

    log_probs = torch.tensor(log_probs).tolist()    
       
    dataset = TensorDataset(torch.tensor(sequences, dtype=torch.int32), torch.tensor(log_probs, dtype=torch.float32))
    data_loader = DataLoader(dataset, batch_size, shuffle=False)

    print('First Iteration:', time.time() - time1)

    data_loader = accelerator.prepare(data_loader)

    with torch.no_grad():
        # In case when interation len is equal 4
        hypothesis_probs = None
        hypothesis_data = None
         
        for i in tqdm(range(4, iter_len)):
            time2 = time.time()
            hypothesis_probs = torch.tensor([], dtype= torch.float32).to(logits.device)
            hypothesis_data = torch.tensor([], dtype=torch.int16).to(logits.device)
            
            for sequences, log_probs in tqdm(data_loader):
                # broad_seq: torch.Size([current_batch_size * vocab_len, seq_len])
                broad_seq = sequences.repeat_interleave(vocab_len, dim=0)

                # broad_probs: torch.Size([current_batch_size * vocab_len])
                broad_probs = log_probs.repeat_interleave(vocab_len)

                current_batch_size = sequences.shape[0]

                # Get only logit vectors from vocab
                logits = model(sequences).logits

                # logits = logits[:, :, vocab]   
                logits = torch.index_select(logits, dim=-1, index=vocab.to(logits.device)) 

                if temp != 1.0:
                    logits = logits / temp

                last_log_probs = torch.log_softmax(logits[:,-1,:], dim=-1)

                # make every token's probability that comes after EOS or PAD to be equal to -0.0 
                last_log_probs[(sequences[:, -1] == EOS) | (sequences[:, -1] == PAD)] = -0.0

                # flat_logits: (current_batch_size * vocab_len)
                flat_logits = last_log_probs.view(-1)
                # del last_log_probs

                # Add next token to each sequence. 
                # Changed this part.
                # new_tokens = torch.tensor([[i for i in range(vocab_len)]]).to(logits.device)
                new_tokens = torch.tensor([[i for i in vocab]]).to(logits.device)
                repeated_tokens = new_tokens.repeat_interleave(current_batch_size, dim=0)

                # after eos token make all tokens eos in a sequence 
                repeated_tokens[(sequences[:, -1] == EOS) | (sequences[:, -1] == PAD)] = PAD

                flat_repeated_tokens = repeated_tokens.view(-1,1).to(logits.device)
                new_sequences = torch.cat((broad_seq, flat_repeated_tokens), 1).to(logits.device)
                # del new_tokens, repeated_tokens, flat_repeated_tokens

                # Pointwise sum. new_probs: torch.Size([current_batch_size * vocab_len])
                new_probs = (flat_logits + broad_probs)
                # del flat_logits, broad_probs
                
                # hypothesis_data: torch.Size([broad_seq.shape + hypothesis_data.shape])
                hypothesis_data = torch.cat((hypothesis_data, new_sequences))

                # hypothesis_probs: torch.Size([new_probs.snew_probshape + hypothesis_probs.shape])
                hypothesis_probs = torch.cat((hypothesis_probs, new_probs))

                # for calculating unique values
                hypothesis_probs = hypothesis_probs.unsqueeze(1).repeat(1, hypothesis_data.shape[-1])
                joined_hypothesis = torch.stack((hypothesis_data, hypothesis_probs), dim=1)

                # make rows unique
                unique_joined_hypothesis = torch.unique_consecutive(joined_hypothesis, dim=0)
                hypothesis_data = unique_joined_hypothesis[:, 0, :].to(torch.int32)
                hypothesis_probs = unique_joined_hypothesis[:, 1, 0]

                if hypothesis_probs.shape[0] > gen_len:
                    sorted_indices = hypothesis_probs.argsort(descending=True)[:gen_len]
                    hypothesis_probs = hypothesis_probs[sorted_indices]
                    hypothesis_data = hypothesis_data[sorted_indices]

            del data_loader

            print(f'Time for iteration {i}:', time.time() - time2)

            # Checking how many lines already have PAD
            mask = hypothesis_data == PAD  
            rows_with_pad = mask.any(dim=1)
            count = rows_with_pad.sum().item()
            print(f"The number of rows that already have PAD is {count} / {len(hypothesis_data)}")  

            if count == len(hypothesis_data):
                print(f"Break at iteration {i}")
                break

            dataset = TensorDataset(hypothesis_data.to(torch.int32), hypothesis_probs)
            data_loader = DataLoader(dataset, batch_size, shuffle=False)
 
            del dataset 
            
        hypothesis_data = hypothesis_data.detach().cpu().tolist()
        hypothesis_probs = hypothesis_probs.detach().cpu().tolist()
    
    return hypothesis_data, hypothesis_probs
  

if __name__ == '__main__':
    args = parse_args()
    manual_seed(1)
    # Tokenizer load
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)   
    print("Get vocab ids...") 

    if args.vocab_path:
        vocab = []  
          
        with open(args.vocab_path, "r") as f:
            for vocab_id in f:
                vocab.append(int(vocab_id))
    else:        
        vocab_ids = sorted(tokenizer.get_vocab().values())

    vocab_ids = sorted(vocab)

    # Model load
    model_load_time = time.time()
    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=args.resume_from_checkpoint, torch_dtype=torch.float16)
    model.eval()

    print(model)
    print('Model loaded:', time.time() - model_load_time)

    beam_decoding_time = time.time()
    accelerator = Accelerator()
    # Beam decoding
    new_hypothesis_data, new_hypothesis_probs = beam_decoding(model=model, accelerator=accelerator, batch_size=args.batch_size, vocab=vocab_ids, iter_len = args.iter_len, gen_len=args.gen_len, temp=args.temperature)

    print("Beam decoding time:", time.time() - beam_decoding_time)

    write_time = time.time()
    print(f"Writing into a file {args.output_path} ...")
    write_tokens_to_csv(args.output_path, new_hypothesis_data, new_hypothesis_probs, tokenizer)

    print("Writing time:", time.time() - write_time)
    print("Overall time:", time.time() - start_time)