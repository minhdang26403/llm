import math

import torch
import torch.nn as nn

from layers.dropout import Dropout
from layers.positional_embedding import RotaryPositionalEmbedding


class MultiheadAttention(nn.Module):
    """
    Causal self-attention with configurable Q/KV head layout.

    This module supports:
    - Standard MHA when ``num_kv_heads == num_heads``
    - GQA when ``1 < num_kv_heads < num_heads``
    - MQA when ``num_kv_heads == 1``

    Args:
        embed_dim: Total dimension of the model.
        d_out: Output embedding dimension across all query heads.
        max_seq_len: Maximum sequence length
        num_heads: Number of parallel attention heads.
        num_kv_heads: Number of key/value heads. If None, defaults to ``num_heads``
        bias: Whether to use bias in input / output projection layers.
    """

    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout_rate: float = 0.0,
        bias: bool = False,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.head_dim = embed_dim // num_heads

        # Number of query heads that share one KV head (for GQA/MQA).
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # K/V use kv_dim, which can be smaller than embed_dim for GQA/MQA,
        # since K/V use less number of heads than Q.
        kv_dim = self.num_kv_heads * self.head_dim

        # Projection layers.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=bias)

        self.dropout = Dropout(dropout_rate)
        self.rope = rope

        # Output projection after concatenating query heads.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.register_buffer(
            "mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1) Project to Q/K/V and split into heads.
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.rope:
            q = self.rope(q)
            k = self.rope(k)

        # 2) If using GQA/MQA, expand KV heads so they align with query heads.
        # For each kv head, repeat it `num_queries_per_kv` times.
        if self.num_queries_per_kv > 1:
            # Note that `repeat_interleave` still allocates memory.
            # Custom kernel can avoid this.
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # 3) Compute scaled dot-product attention scores.
        # Shape: (B, num_heads, N, N)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 4) Apply causal masking (prevent attending to future tokens).
        # Masked positions receive -inf so softmax produces zero probability there.
        # Avoid using `-torch.inf` since the input can be fp16/bfloat16
        causal_mask = self.mask[:seq_len, :seq_len]  # type: ignore
        attn_scores.masked_fill_(~causal_mask, torch.finfo(x.dtype).min)

        # 5) Normalize scores into attention weights.
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6) Multiply attention weights by values.
        # Shape: (B, num_heads, N, head_dim)
        attn_output = attn_weights @ v

        # 7) Recombine heads: transpose and reshape.
        # (B, N, num_heads, head_dim) -> (B, N, E)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        # 8) Final output projection.
        out = self.out_proj(attn_output)

        return out
