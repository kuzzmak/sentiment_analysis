from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import ENCODING


class SentimentDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer, num_samples: int = -1, max_len=128):
        self.data_path = data_path
        self.data = []
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_len = max_len
        
        self._read_data()
        
    def _read_data(self) -> None:
        with open(self.data_path, "r", encoding=ENCODING) as file:
            rows = file.readlines()
            for line in tqdm(rows, "Reading data"):
                # Every row in preprocessed_data.txt is formatted as
                # "<label> <text>"
                label, text = line[:1], line[2:]
                self.data.append((text, int(label)))

        # If num_samples is set, only use the first num_samples
        if self.num_samples > 0:
            self.data = self.data[:self.num_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }