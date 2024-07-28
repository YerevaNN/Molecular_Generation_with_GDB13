import pandas as pd
import argparse

def calc_pred_dup_tp(sum_probs: float, subset_length: int, valid_length: int, gen_length: int) -> int:
    """
    Calculates the predictions for Duplicated TP.

    Args:
        sum_probs (float): The summ of probabilities of validation set.
        subset_length (int): The length of subset.
        valid_length (int): The length of validation set.
        gen_length (int): The length of generation set.

    Returns:
        int: The prediction of Duplicated TP.
    """
    mean_subset_prob = sum_probs/valid_length
    pred_dup_tp_subset = mean_subset_prob*gen_length*subset_length
    pred_tp_dup = round(pred_dup_tp_subset)
    return pred_tp_dup

def calc_pred_uniq_tp(mol_probs: float, subset_length: int, valid_length: int, gen_length: int) -> int:
    """
    Calculates the predictions for Unique TP.

    Args:
        sum_probs (float): The summ of probabilities of validation set.
        subset_length (int): The length of subset.
        valid_length (int): The length of validation set.
        gen_length (int): The length of generation set.

    Returns:
        int: The prediction of Unique TP.
    """
    pred_tp_uniq = round(((1-(1-mol_probs)**gen_length).sum())*subset_length/valid_length)
    return pred_tp_uniq

def parse_args():
    parser = argparse.ArgumentParser(description='Predict statistics on generation')
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="The name of the subset."
    )
    parser.add_argument(
        "--subset_length",
        type=int,
        default=None,
        help="The length of the subset."
    )
    parser.add_argument(
        "--valid_probs_csv",
        type=str,
        default=None,
        help="The path to csv file with validation string's probabilities."
    )
    parser.add_argument(
        "--gen_actual_xlsx",
        type=str,
        default=None,
        help="The path to xlsx file with Actual Duplicated TP and Actual Unique TP."
    )
    parser.add_argument(
        "--gen_length",
        type=int,
        default=None,
        help="The length of generation."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="The path to xlsx file to save the metrics."
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__": 
    args = parse_args()

    SUBSET = args.subset
    SUBSET_LENGTH = args.subset_length 
    # SUBSET_LENGTH = 8_284_280 #aspirin
    # SUBSET_LENGTH = 6_645_440 #sas
    # SUBSET_LENGTH = 5_289_763 #druglike
    # SUBSET_LENGTH = 5_702_826 #eqdist

    VALID_PROBS_CSV = args.valid_probs_csv
    VALID_LENGTH = args.valid_length
    
    OUT_PATH = args.out_path

    GEN_ACTUAL_XLSX = args.gen_actual_xlsx
    GEN_LENGTH = args.gen_length

    probs_df = pd.read_csv(VALID_PROBS_CSV)
    mol_probs = probs_df.groupby('Name')['Probability'].sum()
    sum_probs = mol_probs.sum()
    
    pred_dup_tp = calc_pred_dup_tp(sum_probs, SUBSET_LENGTH, VALID_LENGTH, GEN_LENGTH)
    pred_uniq_tp = calc_pred_uniq_tp(mol_probs, SUBSET_LENGTH, VALID_LENGTH, GEN_LENGTH)
    
    #Actual TP_unique, TP_Duplicates
    actual_metrics = pd.read_excel(GEN_ACTUAL_XLSX)
    actual_dup_tp = actual_metrics['Actual Duplicated TP']
    actual_uniq_tp = actual_metrics['Actual Unique TP']

    diff_dup_tp = abs(pred_dup_tp - actual_dup_tp)
    diff_uniq_tp = abs(pred_uniq_tp - actual_uniq_tp)
    
    metrics = pd.DataFrame({'Subset': SUBSET, "Actual Duplicated TP": actual_dup_tp, 'Predicted Duplicated TP':pred_dup_tp, 'Diff Duplicated TP': diff_dup_tp, "Actual Unique TP": actual_uniq_tp, 'Predicted Unique TP':pred_uniq_tp, 'Diff Unique TP':diff_uniq_tp, "Sum Probs":sum_probs})
    exel_file = metrics.to_excel(OUT_PATH, index=False)