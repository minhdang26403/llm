from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(self, file_path: str | Path, tokenizer: Tokenizer, max_seq_len: int):
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.max_seq_len = max_seq_len
        # PyTorch requires indices into the embedding table to be long integers
        self.token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self) -> int:
        return (len(self.token_ids) - 1) // self.max_seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.max_seq_len
        end = start + self.max_seq_len

        inputs = self.token_ids[start:end]
        targets = self.token_ids[start + 1 : end + 1]

        return inputs, targets
