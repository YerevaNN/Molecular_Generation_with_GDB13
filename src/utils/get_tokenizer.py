from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

def get_tokenizer(tokenizer_path: str, vocab_size: int) -> PreTrainedTokenizerFast:
    """
    Convert a tokenizer file to a Transformer's PreTrainedTokenizerFast

    Args:
        tokenizer_path (str): Path to the tokenizer file.

    Returns:
        PreTrainedTokenizerFast: The prepared tokenizer.
    """
    try:
        # Load the tokenizer from the specified file
        raw_tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
        tokenizer.pad_token = "<pad>"
    except Exception as e:
        raise ValueError(f"Error loading tokenizer from {tokenizer_path}: {e}")

    # Ensure the tokenizer has vocab_size tokens
    try:
        tokenizer_vocab_size = len(tokenizer.get_vocab())
        if tokenizer_vocab_size != vocab_size:
            for i in range(tokenizer_vocab_size, vocab_size):
                symbol = f"madeupword{i:03d}"
                tokenizer.add_tokens(symbol)
        return tokenizer
    except Exception as e:
        raise ValueError(f"Error adjusting tokenizer vocabulary: {e}")