import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer: Tokenizer, max_seq_len):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.max_seq_len = max_seq_len
        # PyTorch requires indices into the embedding table to be long integers
        self.token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return (len(self.token_ids) - 1) // self.max_seq_len

    def __getitem__(self, index):
        start = index * self.max_seq_len
        end = start + self.max_seq_len

        input = self.token_ids[start:end]
        target = self.token_ids[start + 1 : end + 1]

        return input, target
