import pandas as pd
from typing import List
from get_tokenizer import get_tokenizer

def merge_general_stats(
    subsets_list: List[str],
    pretrain: str,
    finetune: str,
    gen_len_str: str,
    valid_length: str,
    str_type: str,
    output_path: str,
    return_df: bool = False
) -> pd.DataFrame:
    """
    Merge general statistics from multiple subsets into a single DataFrame and save to an Excel file.

    Args:
        subsets_list (List[str]): List of subsets to be merged.
        pretrain (str): Pretraining configuration.
        finetune (str): Fine-tuning configuration.
        gen_len_str (str): Generation length as a string.
        valid_length (str): Validation length.
        str_type (str): Type of the statistics.
        output_path (str): Path to save the merged statistics Excel file.
        return_df (bool, optional): Whether to return the DataFrame. Default is False.

    Returns:
        pd.DataFrame: Merged DataFrame if return_df is True, otherwise None.
    """
    merged_df = pd.DataFrame()

    try:
        for subset in subsets_list:
            # Change path if needed.
            path = f'/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13_my/src/statistics/predicted_generation_statistics/{str_type}/{subset}/PRETRAIN_{pretrain}_FINETUNE_{finetune}_GEN_LEN_{gen_len_str}_VALID_LEN_{valid_length}.xlsx'
            subset_df = pd.read_excel(path)
            merged_df = pd.concat([merged_df, subset_df], ignore_index=True)
    except Exception as e:
        print(f"Error merging statistics: {e}")
        return None

    output_path = f'/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13_my/src/statistics/general_statistics/{str_type}/{pretrain}_{finetune}_{gen_len_str}.xlsx'
    try:
        merged_df.to_excel(output_path, index=False)
    except Exception as e:
        print(f"Error saving merged statistics to Excel: {e}")
        return None

    if return_df:
        return merged_df
    return None


def strings_length_stats(tokenizer_path: str, file_path: str, file_extension: str) -> None:
    """
    Calculate and print length statistics of tokenized strings from a file.

    Args:
        tokenizer_path (str): Path to the tokenizer file.
        file_path (str): Path to the input file (CSV or JSONL).
        file_extension (str): The file extension ('csv' or 'jsonl').

    Raises:
        ValueError: If the file extension is unsupported or an error occurs during processing.
    """
    tokenizer = get_tokenizer(tokenizer_path)

    min_val = float('inf')
    max_val = 0
    sum_val = 0
    num_str = 0

    try:
        if file_extension == 'csv':
            data = pd.read_csv(file_path, header=None)[0].to_list()
        elif file_extension == 'jsonl':
            data = pd.read_json(file_path, lines=True)['text'].to_list()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    for sample in data:
        try:
            encoded_sample = tokenizer.tokenize(sample)
            sample_length = len(encoded_sample)

            num_str += 1
            sum_val += sample_length

            if sample_length > max_val:
                max_val = sample_length
            if sample_length < min_val:
                min_val = sample_length
        except Exception as e:
            print(f"Error processing sample '{sample}': {e}")

    if num_str > 0:
        mean_val = sum_val / num_str
        print(f'min: {min_val}, max: {max_val}, mean: {mean_val}')
    else:
        print("No valid samples found.")