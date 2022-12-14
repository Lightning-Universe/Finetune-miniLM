import csv
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


class TextDataset(Dataset):
    def __init__(self, csv_file: str):
        super().__init__()
        csv_file = os.path.expanduser(csv_file)
        with open(csv_file, newline="") as csvfile_handle:
            reader = csv.reader(csvfile_handle, delimiter=",")
            self.rows = list(reader)

    @property
    def labels(self):
        return max(int(r[0]) for r in self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        label, text = self.rows[item]
        label = int(label) - 1
        return dict(text=text, label=label)


class TokenizedDataloader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_token_len: int = 256,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = TextEncodingCollate(tokenizer, max_token_len)
        num_workers = num_workers if num_workers is not None else os.cpu_count()
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)


class TextEncodingCollate:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_sequence_length=256):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, samples):
        return self.tokenize(samples), self.labels(samples)

    def tokenize(self, samples):
        texts = [sample["text"] for sample in samples]
        return self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_length
        )

    def labels(self, samples):
        return torch.tensor([sample["label"] for sample in samples])
