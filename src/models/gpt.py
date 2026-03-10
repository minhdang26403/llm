import torch
import torch.nn as nn

from layers.attention import MultiheadAttention
from layers.dropout import Dropout
from layers.norm import LayerNorm
from layers.positional_embedding import SinusoidalEmbedding


class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = LayerNorm(config.embed_dim)
        self.attn = MultiheadAttention(
            config.embed_dim,
            config.max_seq_len,
            config.num_heads,
            config.num_kv_heads,
        )
        self.ln_2 = LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim, bias=False),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.embed_dim, bias=False),
        )
        self.dropout = Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Communication Phase (Attention)
        # We use Pre-Norm architecture here
        x = self.ln_1(x)
        x = x + self.dropout(self.attn(x))

        # 2. Computation Phase (MLP)
        x = self.ln_2(x)
        x = x + self.dropout(self.mlp(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = SinusoidalEmbedding(
            config.embed_dim, config.max_seq_len
        )
        self.gpt_blocks = nn.ModuleList(
            GPTBlock(config) for _ in range(config.num_blocks)
        )
        self.ln = LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # We use weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        x = x + self.positional_embedding(x)

        for block in self.gpt_blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.lm_head(x)

        return logits
