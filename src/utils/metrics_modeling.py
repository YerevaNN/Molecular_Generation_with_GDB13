import time
import argparse
import pandas as pd

def calc_pred_dup_tp(sum_probs: float, subset_length: int, valid_length: int, gen_length: int) -> int:
    """
    Calculates the predictions for Duplicated TP.

    Args:
        sum_probs (float): The sum of probabilities of validation set.
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
        sum_probs (float): The sum of probabilities of validation set.
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
        "--valid_length",
        type=int,
        default=10_000,
        help="The length of validation set."
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

    VALID_PROBS_CSV = args.valid_probs_csv
    VALID_LENGTH = args.valid_length
    
    OUT_PATH = args.out_path

    GEN_ACTUAL_XLSX = args.gen_actual_xlsx
    GEN_LENGTH = args.gen_length

    time1 = time.time()
    probs_df = pd.read_csv(VALID_PROBS_CSV)
    mol_probs = probs_df.groupby('Name')['Probability'].sum()[:VALID_LENGTH]

    sum_probs = mol_probs.sum()
    df = pd.DataFrame(columns=["Recall"])
    
    pred_dup_tp = calc_pred_dup_tp(sum_probs, SUBSET_LENGTH, VALID_LENGTH, GEN_LENGTH)
    for i, GEN_LENGTH in enumerate([1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000,4500000,5000000,5500000,6000000,6500000,7000000,7500000,8000000,8500000, \
                       9000000,9500000,10000000,20000000,30000000,40000000,50000000,60000000,70000000,80000000,90000000,100000000,200000000,300000000,\
                       400000000,500000000,600000000,700000000,800000000,900000000,1000000000]):
        
        pred_uniq_tp = calc_pred_uniq_tp(mol_probs, SUBSET_LENGTH, VALID_LENGTH, GEN_LENGTH)
        # print('Metrics are calculated.')
        # print('Validation length:', VALID_LENGTH)
        # print('Time:', time.time()-time1)

        # TP_unique, TP
        actual_metrics = pd.read_excel(GEN_ACTUAL_XLSX)
        # actual_dup_tp = actual_metrics['TP']
        actual_uniq_tp = actual_metrics['Unique TP']

        # diff_dup_tp = abs(pred_dup_tp - actual_dup_tp)
        diff_uniq_tp = abs(pred_uniq_tp - actual_uniq_tp)
    
        # metrics = pd.DataFrame({'Subset': SUBSET, "TP": actual_dup_tp, 'Predicted TP':pred_dup_tp, 'Diff TP': diff_dup_tp, "Unique TP": actual_uniq_tp, 'Predicted Unique TP':pred_uniq_tp, 'Diff Unique TP':diff_uniq_tp, "Validation Length":VALID_LENGTH, "Sum Probs":sum_probs})
        # metrics = pd.DataFrame({"GEN_LENGTH":GEN_LENGTH,'Predicted Unique TP':pred_uniq_tp / 8284280 * 100})
        # print("GEN_LENGTH",GEN_LENGTH,'Predicted Unique TP',pred_uniq_tp / 8284280 * 100)

        df.loc[len(df)] = [pred_uniq_tp/8284280 *100]
        
    print(df)    
    exel_file = df.to_csv(OUT_PATH, index=False)