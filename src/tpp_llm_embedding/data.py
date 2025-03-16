"""
Dataset and Data Loader for the TPP-LLM
"""
import json
import os
import os.path
import random
from typing import List

import torch
from torch.utils.data import Dataset

torch.manual_seed(0)


class TPPLLMDataset(Dataset):
    """
    TPP-LLM Dataset
    """

    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: dict) -> dict:
    """
    Data collation function for the data loader

    :param batch: items in a batch
    :return: batched data
    """
    return {
        'time_since_start': [torch.FloatTensor(item['time_since_start']) for item in batch],
        'time_since_last_event': [torch.FloatTensor(item['time_since_last_event']) for item in batch],
        'type_event': [torch.LongTensor(item['type_event']) for item in batch],
        'type_text': [item['type_text'] for item in batch],
        'description': [item['description'] for item in batch],
    }


def merge_datasets(source_folders: List[str], target_folder: str, sample_ratio: float, seed=0) -> None:
    """
    Merge datasets into a multi-domain dataset

    :param source_folders: a list of source folders
    :param target_folder: a target folder
    :param sample_ratio: sample ratio
    :param seed: seed for reproducibility
    """
    random.seed(seed)

    json_files = ['train.json', 'dev.json', 'test.json']
    merged_data = {file_name: [] for file_name in json_files}

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Iterate through all source folders
    for folder in source_folders:
        dataset_name = os.path.basename(folder)
        for file_name in json_files:
            file_path = os.path.join(folder, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    sample_size = int(len(data) * sample_ratio)
                    sample_data = random.sample(data, sample_size)
                    merged_data[file_name].extend(
                        [{'dataset': dataset_name, **entry} for entry in sample_data])
                    print(f'{file_path}: {len(data)} sequences -> {len(sample_data)} sequences')

    # Write the merged data to the target folder
    for file_name, data in merged_data.items():
        target_file_path = os.path.join(target_folder, file_name)
        with open(target_file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f'{target_file_path}: {len(data)} sequences')

    print(f"Merged datasets saved to {target_folder}")
