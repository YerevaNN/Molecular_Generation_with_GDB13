from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

def get_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """
    Convert a tokenizer file to a Transformer's PreTrainedTokenizerFast and ensure it has 192 tokens.

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

    # Ensure the tokenizer has 192 tokens
    try:
        vocab_size = len(tokenizer.get_vocab())
        if vocab_size != 192:
            for i in range(vocab_size, 192):
                symbol = f"madeupword{i:03d}"
                tokenizer.add_tokens(symbol)
        return tokenizer
    except Exception as e:
        raise ValueError(f"Error adjusting tokenizer vocabulary: {e}")