import json
import torch 
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import TensorDataset, DataLoader

def get_tokenized_data(path_to_data, tokenizer):
    """
    Loads and tokenizes data from the specified path.

    Args:
        path_to_data (str): The path to the data file.
        tokenizer (transformers.Tokenizer): The tokenizer to use for encoding.

    Returns:
        tuple: A tuple containing the padded input and target tensors.
    """
    inputs = []
    with open(path_to_data, "r") as file:
        for line_str in tqdm(file):
            line_obj = json.loads(line_str)
            sample = line_obj["text"]
            
            encoded_sample = tokenizer.encode(sample, add_special_tokens=True)
            inputs.append(encoded_sample)
            
    padded_inputs = pad_sequence([torch.tensor(seq) for seq in inputs], batch_first=True, padding_value=tokenizer.pad_token_id)
    return padded_inputs
 

def to_dataloader(path_to_data, tokenizer, batch_size):
    """
    Creates a DataLoader for the tokenized data.

    Args:
        path_to_data (str): The path to the data file.
        tokenizer (transformers.Tokenizer): The tokenizer to use for encoding.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        DataLoader: A DataLoader instance for the tokenized data.
    """
    inputs  = get_tokenized_data(path_to_data, tokenizer)
    dataset = TensorDataset(inputs)
    data_loader = DataLoader(dataset, batch_size, shuffle=False)
    return data_loader