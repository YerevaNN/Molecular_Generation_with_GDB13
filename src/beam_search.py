import os
import csv
import time
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from transformers import OPTForCausalLM

from utils.get_tokenizer import get_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate strings with beam search.")
    parser.add_argument(
        "--tokenizer_path",
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
        "--batch_size",
        type=int,
        default=1,
        help="Batch size of the validation data.",
    )
    parser.add_argument(
        "--output_beams",
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
        default=32,
        help="The length of iterations."
    )
    args = parser.parse_args()

    return args

def collect_vocab_ids(tokenizer):
    vocab_dict = tokenizer.get_vocab()
    vocab_ids_list = []

    for i in vocab_dict:
        vocab_ids_list.append(tokenizer.encode(i)[:2])

    return vocab_ids_list


def write_tokens_to_csv(hypothesis_tokens, tokenizer, filename):
  """
  Writes tokenized hypotheses and their probabilities to a CSV file.

  Args:
      hypothesis_tokens (torch.Tensor): Tensor of hypothesis token IDs (shape: (batch_size, sequence_length)).
      hypothesis_probs (torch.Tensor): Tensor of hypothesis probabilities (shape: (batch_size,)).
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for converting IDs to strings.
      filename (str): Name of the CSV file to write to.
  """

  with open(filename, 'w', newline='') as csvfile:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    writer = csv.writer(csvfile)
    writer.writerow(['Hypothesis'])

    # Iterate over each hypothesis
    for tokens in hypothesis_tokens:
      hypothesis_text = tokenizer.decode(tokens, skip_special_tokens=True)
      writer.writerow([hypothesis_text])  


def to_data_loader(vocab, batch_size):
    inputs = []
    targets = []
    sequences = []

    for token_1 in vocab:
        for token_2 in vocab:
            inputs.append([0, token_1])
            targets.append([token_1, token_2])
            sequences.append([0,token_1, token_2])

    dataset = TensorDataset(torch.tensor(inputs), torch.tensor(targets))
    data_loader = DataLoader(dataset, batch_size, shuffle=False)
    return data_loader, sequences


@torch.no_grad()
def next_iteration(model, batch_size, vocab, iter_len, gen_len):
    model.eval()

    time1 = time.time()
    log_probs = []
    data_loader, sequences = to_data_loader(vocab, batch_size=batch_size)

    for input, target in data_loader:
        input = input.to(model.device)
        target = target.to(model.device)
        logits = model(input).logits
        flatten_logits = logits.view(-1,logits.size(-1))
        targets = target.view(-1)

        cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        flat_log_probs = (-cross_entropy(flatten_logits, targets))
        log_probs = log_probs + (flat_log_probs.view(-1,2).sum(dim=-1)).tolist()

    dataset = TensorDataset(torch.tensor(sequences, dtype=torch.int32), torch.tensor(log_probs, dtype=torch.float32))
    data_loader = (DataLoader(dataset, batch_size, shuffle=False))

    print('First Iteration:', time.time() - time1)

    device = 'cpu'

    with torch.no_grad():
        for i in range(4,iter_len):
            time2 = time.time()
            hypothesis_probs = torch.tensor([], dtype= torch.float32, device=device)
            hypothesis_data = torch.tensor([], dtype=torch.int16, device=device)
            
            for sequences, log_probs in data_loader:
                # broad_seq: torch.Size([current_batch_size * 192, seq_len])
                broad_seq = (sequences.repeat_interleave(192,dim=0)).to(device)

                # broad_probs: torch.Size([current_batch_size * 192])
                broad_probs = (log_probs.repeat_interleave(192)).to(device)

                current_batch_size = sequences.shape[0]

                sequences = sequences.to(model.device)

                if TEMPERATURE !=1.0:
                    logits = (model(sequences).logits)/TEMPERATURE
                else:
                    logits = (model(sequences).logits)
                del sequences 

                last_log_probs = (torch.log_softmax(logits[:,-1,:], dim=-1)).to(device)

                # flat_logits: (current_batch_size * 192)
                flat_logits = last_log_probs.view(-1)
                del last_log_probs

                # Add next token to each sequence. 
                new_tokens = torch.tensor([[i for i in range(192)]])
                repeated_tokens = new_tokens.repeat_interleave(current_batch_size,dim=0)
                flat_repeated_tokens = repeated_tokens.view(-1,1)
                new_sequences = torch.cat((broad_seq,flat_repeated_tokens),1)
                del new_tokens, repeated_tokens, flat_repeated_tokens

                # Pointwise sum. new_probs: torch.Size([current_batch_size * 192])
                new_probs = (flat_logits + broad_probs)
                del flat_logits, broad_probs
                
                # hypothesis_data: torch.Size([broad_seq.shape + hypothesis_data.shape])
                hypothesis_data = torch.cat((hypothesis_data, new_sequences))

                # hypothesis_probs: torch.Size([new_probs.shape + hypothesis_probs.shape])
                hypothesis_probs = torch.cat((hypothesis_probs, new_probs))

                if hypothesis_probs.shape[0]>gen_len:
                    sorted_indices = hypothesis_probs.argsort(descending=True)[:gen_len]
                    hypothesis_probs = hypothesis_probs[sorted_indices]
                    hypothesis_data = hypothesis_data[sorted_indices]


            # sorted_indices = hypothesis_probs.argsort(descending=True)[:gen_len]
            # new_hypothesis_probs = hypothesis_probs[sorted_indices]
            # new_hypothesis_data = hypothesis_data[sorted_indices]
            del data_loader 

            dataset = TensorDataset(hypothesis_data.to(torch.int32), hypothesis_probs)
            data_loader = (DataLoader(dataset, batch_size, shuffle=False))
            del dataset 
            
            print(f'Time for iteration {i}:', time.time() - time2)
    
    return hypothesis_data, hypothesis_probs
  
if __name__ == '__main__':
    args = parse_args()

    tokenizer_path = args.tokenizer_path
    checkpoint = args.resume_from_checkpoint
    batch_size = args.batch_size
    output_beams = args.output_beams
    gen_len = args.gen_len
    iter_len = args.iter_len
    TEMPERATURE = args.temperature

    tokenizer = get_tokenizer(tokenizer_path=tokenizer_path)

    time1 = time.time()
    vocab_ids_list = collect_vocab_ids(tokenizer) # The ids of vocabulary tokens
    print('Vocab Collection done:', time.time() - time1)

    time2 = time.time()
    model = OPTForCausalLM.from_pretrained(pretrained_model_name_or_path=checkpoint,torch_dtype=torch.float16).to('cuda')
    print('Model loaded:', time.time() - time2)

    time3 = time.time()
    new_hypothesis_data, new_hypothesis_probs = next_iteration(model=model, batch_size=batch_size, vocab=vocab_ids_list, iter_len = iter_len, gen_len=gen_len)

    write_tokens_to_csv(new_hypothesis_data, new_hypothesis_probs, tokenizer, output_beams)