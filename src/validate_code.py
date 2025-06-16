import os 
import sys
import csv
import ast
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
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM, AutoTokenizer

from utils.get_tokenizer import get_tokenizer
from utils.data_preprocessing import to_dataloader



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def top_k_sampling(logits: Tensor, k: int) -> Tensor:
    """
    Apply top-k sampling on the logits.

    Args:
        logits (Tensor): A tensor of logits with shape [batch_size, seq_len, vocab_size].
        k (int): The number of top elements to keep for sampling.

    Returns:
        Tensor: A tensor of the same shape as `logits` with modified probabilities after top-k sampling.
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Get the top-k probabilities and their indices
    topk_probs, topk_indices = probs.topk(k=k, dim=-1, largest=True, sorted=False)

    # Create a binary mask for top-k indices
    mask = torch.zeros_like(probs).scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_probs))

    # Apply the mask and normalize the probabilities
    masked_probs = mask * probs
    row_sums = masked_probs.sum(dim=-1, keepdim=True)
    masked_probs = masked_probs / row_sums

    return masked_probs


def top_p_sampling(logits: Tensor, p: float = 1.0) -> Tensor:
    """
    Apply top-p (nucleus) sampling on the logits.

    Args:
        logits (Tensor): A tensor of logits with shape [batch_size, seq_len, vocab_size].
        p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        Tensor: A tensor of the same shape as `logits` with modified probabilities after top-p sampling .
    """

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)  

    # Sort the probabilities in descending order and get the sorted indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Calculate cumulative probabilities
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    
    # Create a mask for probabilities where cumulative probability exceeds p
    # Ensure at least one token is kept
    mask = cumulative_probs <= p
    mask[..., 1:] = mask[..., :-1]  
    mask[..., 0] = True  
    
    # Apply the mask and normalize probabilities
    masked_probs = torch.where(mask, sorted_probs, torch.tensor(0.0, device=sorted_probs.device))
    masked_probs /= masked_probs.sum(dim=-1, keepdim=True)
    
    # Reconstruct the tensor in the original order
    top_tensor = torch.zeros_like(probs)
    top_tensor.scatter_(dim=-1, index=sorted_indices, src=masked_probs)

    return top_tensor    


class CanonicalizeCommutativeOps(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, (ast.Add, ast.Mult)):
            operands = self.flatten_commutative_chain(node, type(node.op))
            sorted_operands = sorted(operands, key=self.sort_key)

            return self.build_commutative_chain(sorted_operands, node.op)

        return node

    def flatten_commutative_chain(self, node, op_type):
        if isinstance(node, ast.BinOp) and isinstance(node.op, op_type):

            return self.flatten_commutative_chain(node.left, op_type) + \
                   self.flatten_commutative_chain(node.right, op_type)
        else:
            return [node]

    def build_commutative_chain(self, operands, op):
        if not operands:
            return ast.Constant(value=0 if isinstance(op, ast.Add) else 1)

        expr = operands[0]

        for operand in operands[1:]:
            expr = ast.BinOp(left=expr, op=op, right=operand)

        return expr

    def sort_key(self, node):
        if isinstance(node, ast.Constant):
            return (0, repr(node.value))
        elif isinstance(node, ast.Name):
            return (1, node.id)
        else:
            return (2, ast.dump(node))


def normalize_code(code):
    tree = ast.parse(code)
    normalized = CanonicalizeCommutativeOps().visit(tree)
    ast.fix_missing_locations(normalized)

    return ast.unparse(normalized)


def get_bytecode(code_str):
    """Compile code string to bytecode."""
    code = ast.parse(code_str)
    
    # Bytecode Object
    bytecode = compile(code, '<string>', 'exec')

    # Bytecode bytes
    bytecode = bytecode.co_code  

    return bytecode


@torch.no_grad()
def calc_sequence_prob(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    inputs: Union[str, torch.Tensor],
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float]
) -> torch.Tensor:
    """
    Calculate the probability of a sequence given a model, tokenizer and decoding strategy.

    Args:
        model: The language model used for generating logits.
        tokenizer: The tokenizer for encoding and decoding sequences.
        inputs: Tensor of input sequences (shape: [batch_size, seq_length]).
        targets: Tensor of target sequences (shape: [batch_size, seq_length]).
        temperature: Temperature parameter for scaling logits.
        k: Top-k sampling parameter.
        p: Top-p sampling parameter.

    Returns:
        List of lists where each sublist contains a string representation of the sequence and its probability.
    """
    model.eval()

    vocab_size = len(tokenizer.get_vocab())

    targets = inputs[0][..., 1:]
    inputs = inputs[0][..., :-1]

    # Decode target sequence
    # Convert to canonical string if specified
    target_seq = []
    print("Target shape", targets.shape)
    for sequence_ids in tqdm(targets):
        decoded_seq = tokenizer.decode(sequence_ids, skip_special_tokens=True)

        # Make canonial
        normalized_line = normalize_code(decoded_seq)
        bytecode_line = get_bytecode(normalized_line)

        target_seq.append(str(bytecode_line))

    # Prepare logits and targets
    print("Calculating logits ...")
    logits = (model(inputs).logits)             # shape: [batch_size, seq_length, vocab_size]
    logits = logits.view(-1, logits.size(-1))   # shape: [batch_size * seq_length, vocab_size]
    targets = targets.contiguous().view(-1)     # shape: [batch_size * seq_length]

    # Get modified (wrt sampling strategy) probabilities
    if temperature != 1.0:
        logits /= temperature

    if top_k < vocab_size:
        probs = top_k_sampling(logits, top_k)
    elif top_p < 1.0:
        probs = top_p_sampling(logits, top_p)
    else: 
        probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs)        # shape: [batch_size * seq_length, vocab_size]

    # Calculate probabilities 
    loss = nn.NLLLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    log_token_probs = -loss(log_probs, targets)
    log_seq_probs = log_token_probs.view(inputs.size(0), -1)
    log_seq_probs = log_seq_probs.sum(-1)

    seq_probs = torch.exp((log_seq_probs)).tolist()  

    rows = [[str_repr, prob] for str_repr, prob in zip(target_seq, seq_probs)] 

    return rows


def write_probs_csv(
    output_path: str,
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    data_loader: Any,
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float]
) -> None:

    # if os.path.exists(output_path):
    #     print(f"Error: {output_path} already exists.")
    #     sys.exit(1)

    with open(output_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Probability"])
        
        print("Calculating probabilities ...")
        for inputs in tqdm(data_loader):
            probabilities = calc_sequence_prob(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )
            writer.writerows(probabilities)

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

    accelerator = Accelerator()

    data_path = args.data_path
    tokenizer_path = args.tokenizer_path
    checkpoint = args.resume_from_checkpoint
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    output_path = args.output_path
    batch_size = args.batch_size

    # Prepare the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)   
        
    # Get the dataloader
    time1 = time.time()
    # 14724
    tokenizer.pad_token_id = 128001
    print("fisrt", tokenizer.pad_token_id)

    # Isn't shuffled
    data_loader = to_dataloader(path_to_data=data_path, tokenizer=tokenizer, batch_size=batch_size)
    print("Time for data creating:", time.time()-time1)

    # Get the model
    time1 = time.time()
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=checkpoint)
    print("Time for forward:", time.time()-time1)

    # Prepare the dataloader and the model
    time1 = time.time()
    model, data_loader = accelerator.prepare(model, data_loader)
    print("Time for accelerator:", time.time()-time1)
    
    # Write probabilities in the csv
    time1 = time.time()
    write_probs_csv(output_path=output_path, model=model, tokenizer=tokenizer, data_loader=data_loader, top_k=top_k, top_p=top_p, temperature=temperature) 
    print("Time for writing:", time.time()-time1)