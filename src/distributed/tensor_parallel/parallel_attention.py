import math

import torch
import torch.distributed as dist
import torch.nn as nn

from layers.dropout import Dropout
from layers.positional_embedding import RotaryPositionalEmbedding

from .parallel_linear import ColumnParallelLinear, RowParallelLinear


class ParallelAttention(nn.Module):
    mask: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor

    def __init__(
        self,
        max_seq_len: int,
        embed_dim: int,
        head_dim: int,
        tp_group: dist.ProcessGroup,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout_rate: float = 0.0,
        bias: bool = False,
        rope: RotaryPositionalEmbedding | None = None,
        use_cache: bool = False,
    ):
        super().__init__()
        self.tp_group = tp_group

        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.global_num_heads = num_heads
        self.global_num_kv_heads = num_kv_heads if num_kv_heads else num_heads

        world_size = dist.get_world_size(tp_group)

        if self.global_num_heads % world_size != 0:
            raise ValueError(
                f"Global number of heads ({self.global_num_heads}) must be divisible "
                f"by number of tensor parallel processes ({world_size})"
            )
        if self.global_num_kv_heads % world_size != 0:
            raise ValueError(
                f"Global number of KV heads ({self.global_num_kv_heads}) must be "
                f"divisible by number of tensor parallel processes ({world_size})"
            )

        self.local_num_heads = self.global_num_heads // world_size
        self.local_num_kv_heads = self.global_num_kv_heads // world_size

        # Number of query heads that share one KV head (for GQA/MQA).
        self.num_queries_per_kv = self.global_num_heads // self.global_num_kv_heads

        # K/V use kv_dim, which can be smaller than q_dim for GQA/MQA,
        # since K/V use less number of heads than Q.
        self.global_q_dim = self.global_num_heads * self.head_dim
        global_kv_dim = self.global_num_kv_heads * self.head_dim

        # Projection layers.
        self.q_proj = ColumnParallelLinear(embed_dim, self.global_q_dim, self.tp_group)
        self.k_proj = ColumnParallelLinear(embed_dim, global_kv_dim, self.tp_group)
        self.v_proj = ColumnParallelLinear(embed_dim, global_kv_dim, self.tp_group)

        self.rope = rope
        self.dropout = Dropout(dropout_rate)

        # Output projection after concatenating query heads.
        self.out_proj = RowParallelLinear(self.global_q_dim, embed_dim, self.tp_group)

        self.register_buffer(
            "mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        )

        self.use_cache = use_cache
        self.cache_len = 0
        if use_cache:
            # In training, we disable the KV cache, so these buffers are not saved in
            # the model checkpoint.
            # Set `persistent` to False to prevent PyTorch from trying to look for this
            # buffer in the model checkpoint file.
            self.register_buffer(
                "k_cache",
                torch.zeros(1, self.local_num_kv_heads, max_seq_len, self.head_dim),
                persistent=False,
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(1, self.local_num_kv_heads, max_seq_len, self.head_dim),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q: torch.Tensor = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.local_num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k: torch.Tensor = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.local_num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v: torch.Tensor = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.local_num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.rope:
            q = self.rope(q, self.cache_len)
            k = self.rope(k, self.cache_len)

        if self.use_cache:
            self.k_cache[:, :, self.cache_len : self.cache_len + seq_len, :] = k
            self.v_cache[:, :, self.cache_len : self.cache_len + seq_len, :] = v
            self.cache_len += seq_len
            k = self.k_cache[:, :, : self.cache_len, :]
            v = self.v_cache[:, :, : self.cache_len, :]

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if seq_len > 0:
            causal_mask = self.mask[:seq_len, :seq_len]
            attn_scores.masked_fill_(~causal_mask, torch.finfo(x.dtype).min)

        attn_weights: torch.Tensor = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ v
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.local_num_heads * self.head_dim)
        )

        return self.out_proj(attn_output)

    def reset_cache(self) -> None:
        self.k_cache.fill_(0)
        self.v_cache.fill_(0)
        self.cache_len = 0
