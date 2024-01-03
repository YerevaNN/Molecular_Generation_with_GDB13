import os
import sys
import random
import json
import re
import glob
import shutil


def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    return lines


def save_train_file(file_path, train_data):

    folder_path = os.path.dirname(file_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(file_path, 'w') as file:
        for line in train_data:
            file.write(line)


def save_valid_file(file_path, subset_path):  
    folder_path = os.path.dirname(subset_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    shutil.copyfile(file_path, subset_path)       



if __name__ == "__main__":
    subset_name = "equal_dist_sf" 

    train_file_path = glob.glob(f"./data_bin_{subset_name}_1000K/train/00/*.jsonl")[0]
    valid_file_path = glob.glob(f"./data_bin_{subset_name}_1000K/valid/00/*.jsonl")[0]

    subsets = {
        4000: '4K', 16000: '16K',
        64000: '64K', 256000: '256K'
    }

    train_data = read_file(train_file_path)

    for num_samples, name in subsets.items():
        # train
        train_subset_path = f"./data_bin_{subset_name}_{name}/train/00/train_{subset_name}.jsonl"
        data_subset = train_data[:num_samples]
        save_train_file(train_subset_path, data_subset)

        # valid
        valid_subset_path = f"./data_bin_{subset_name}_{name}/valid/00/valid_{subset_name}.jsonl"
        save_valid_file(valid_file_path, valid_subset_path) 