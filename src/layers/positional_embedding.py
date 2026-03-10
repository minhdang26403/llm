import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()

        t = torch.arange(0, embed_dim, 2)
        inv_freq = 10000 ** (-t / embed_dim)
        position = torch.arange(max_seq_len)
        out = position.unsqueeze(1) * inv_freq  # Shape: (seq_len, d/2)

        # Optimized Interleaving: (seq_len, d/2, 2) -> (seq_len, d)
        # This is often faster than slice assignment [:, 0::2]
        embedding = torch.stack([out.sin(), out.cos()], dim=-1).flatten(1)
        self.register_buffer("embedding", embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return self.embedding[:seq_len, :]
