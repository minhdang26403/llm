from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, file_path: Path, max_seq_len: int):
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        # Numpy memmap handles the file descriptor and byte-translation automatically!
        # It treats the binary file as a massive array of 32-bit integers.
        self.data = np.memmap(file_path, dtype=np.uint32, mode="r")
        self.max_seq_len = max_seq_len
        self.num_elements = (len(self.data) - 1) // self.max_seq_len

    def __len__(self) -> int:
        return self.num_elements

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.max_seq_len
        end = start + self.max_seq_len + 1

        # Slice the numpy array (this is a zero-copy operation, incredibly fast)
        chunk = self.data[start:end]

        # PyTorch's CrossEntropyLoss and Embedding requires int64 (torch.long)
        # We cast the numpy array to int64, then convert to a tensor
        chunk_tensor = torch.tensor(chunk.astype(np.int64), dtype=torch.long)

        inputs = chunk_tensor[:-1]
        targets = chunk_tensor[1:]

        return inputs, targets
