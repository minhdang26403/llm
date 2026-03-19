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
        use_cache: Whether to enable KV cache for this layer or not
    """

    mask: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout_rate: float = 0.0,
        bias: bool = False,
        rope: RotaryPositionalEmbedding | None = None,
        use_cache: bool = False,
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

        self.use_cache = use_cache
        self.cache_len = 0
        if use_cache:
            # In training, we disable the KV cache, so these buffers are not saved in
            # the model checkpoint.
            # Set `persistent` to False to prevent PyTorch from trying to look for this
            # buffer in the model checkpoint file.
            self.register_buffer(
                "k_cache",
                torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim),
                persistent=False,
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim),
                persistent=False,
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
            q = self.rope(q, self.cache_len)
            k = self.rope(k, self.cache_len)

        if self.use_cache:
            self.k_cache[:, :, self.cache_len : self.cache_len + seq_len, :] = k
            self.v_cache[:, :, self.cache_len : self.cache_len + seq_len, :] = v
            self.cache_len += seq_len
            k = self.k_cache[:, :, : self.cache_len, :]
            v = self.v_cache[:, :, : self.cache_len, :]

        # 2) If using GQA/MQA, expand KV heads so they align with query heads.
        # For each kv head, repeat it `num_queries_per_kv` times.
        if self.num_queries_per_kv > 1:
            # Note that `repeat_interleave` still allocates memory.
            # Custom kernel can avoid this.
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # 3) Compute scaled dot-product attention scores.
        # Shape: (B, num_heads, Nq, Nk)
        # In decode phase: Nq = 1 while Nk is the length of the current sentence
        attn_scores: torch.Tensor = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 4) Apply causal masking (prevent attending to future tokens).
        # Masked positions receive -inf so softmax produces zero probability there.
        # Avoid using `-torch.inf` since the input can be fp16/bfloat16
        if seq_len > 1:
            causal_mask = self.mask[:seq_len, :seq_len]
            attn_scores.masked_fill_(~causal_mask, torch.finfo(x.dtype).min)

        # 5) Normalize scores into attention weights.
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6) Multiply attention weights by values.
        # Shape: (B, num_heads, Nq, head_dim)
        attn_output: torch.Tensor = attn_weights @ v

        # 7) Recombine heads: transpose and reshape.
        # (B, Nq, num_heads, head_dim) -> (B, Nq, E)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        # 8) Final output projection.
        out = self.out_proj(attn_output)

        return out

    def reset_cache(self) -> None:
        self.k_cache.fill_(0)
        self.v_cache.fill_(0)
        self.cache_len = 0


class MultiheadLatentAttention(nn.Module):
    mask: torch.Tensor
    c_kv_cache: torch.Tensor
    k_rope_cache: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        latent_dim: int,
        max_seq_len: int,
        dropout_rate: float = 0.0,
        bias: bool = False,
        rope: RotaryPositionalEmbedding | None = None,
        rope_dim: int | None = None,
        use_cache: bool = False,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query projection matrix
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Compression and Decompression matrices
        self.latent_dim = latent_dim
        self.down_kv_proj = nn.Linear(embed_dim, latent_dim, bias=bias)
        self.up_k_proj = nn.Linear(latent_dim, embed_dim, bias=bias)
        self.up_v_proj = nn.Linear(latent_dim, embed_dim, bias=bias)

        # Optional dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Mask for causal attention
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        )

        # Rotary Embedding support
        self.rope = rope
        if self.rope:
            assert rope_dim is not None
            if rope_dim % num_heads != 0:
                raise ValueError("rope_dim must be divisible by num_heads")

            self.rope_head_dim = rope_dim // num_heads
            self.q_rope_proj = nn.Linear(embed_dim, rope_dim, bias=bias)
            self.k_rope_proj = nn.Linear(embed_dim, rope_dim, bias=bias)

        self.use_cache = use_cache
        if self.use_cache:
            self.cache_len = 0
            self.register_buffer(
                "c_kv_cache",
                torch.zeros(1, max_seq_len, self.latent_dim),
                persistent=False,
            )
            self.register_buffer(
                "k_rope_cache",
                torch.zeros(1, self.num_heads, max_seq_len, self.rope_head_dim),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Perform Q projection as usual
        # Shape: (B, nh, N, hd)
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compress token embedding into a latent vector
        # Shape: (B, N, l)
        c_kv: torch.Tensor = self.down_kv_proj(x)

        if self.use_cache:
            self.c_kv_cache[:, self.cache_len : self.cache_len + seq_len, :] = c_kv
            c_kv = self.c_kv_cache[:, : self.cache_len + seq_len, :]

        if self.training:
            # Decompress the latent vector into K and V
            # Shape: (B, nh, N, hd)
            k: torch.Tensor = (
                self.up_k_proj(c_kv)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            # Shape: (B, nh, N, hd)
            v: torch.Tensor = (
                self.up_v_proj(c_kv)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            # Shape: (B, nh, N, N)
            attn_scores = q @ k.transpose(-2, -1)
        else:
            # Shape: (B, nh, N, l)
            q_compressed: torch.Tensor = q @ self.up_k_proj.weight.view(
                1, self.num_heads, self.head_dim, self.latent_dim
            )
            attn_scores = q_compressed @ c_kv.view(
                batch_size, 1, -1, self.latent_dim
            ).transpose(-2, -1)

        total_dim = self.head_dim
        if self.rope:
            # Project the token embeddings into small vectors that represent
            # positional information
            q_rope: torch.Tensor = (
                self.q_rope_proj(x)
                .view(batch_size, seq_len, self.num_heads, self.rope_head_dim)
                .transpose(1, 2)
            )
            k_rope = (
                self.k_rope_proj(x)
                .view(batch_size, seq_len, self.num_heads, self.rope_head_dim)
                .transpose(1, 2)
            )

            # Apply the Rotary Embedding
            q_rope = self.rope(q_rope, self.cache_len)
            k_rope = self.rope(k_rope, self.cache_len)

            if self.use_cache:
                self.k_rope_cache[
                    :, :, self.cache_len : self.cache_len + seq_len, :
                ] = k_rope
                k_rope = self.k_rope_cache[:, :, : self.cache_len + seq_len, :]

            total_dim += self.rope_head_dim
            attn_scores += q_rope @ k_rope.transpose(-2, -1)

        attn_scores /= math.sqrt(total_dim)

        if self.use_cache:
            self.cache_len += seq_len

        # No need to apply mask if we have only one token
        if seq_len > 1:
            causal_mask = self.mask[:seq_len, :seq_len]
            attn_scores.masked_fill_(~causal_mask, torch.finfo(x.dtype).min)

        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        if self.training:
            # Shape: (B, nh, N, hd)
            attn_output: torch.Tensor = attn_weights @ v

            # Shape: (B, N, E)
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, self.embed_dim)
            )

            out = self.out_proj(attn_output)
        else:
            # Shape: (B, nh, N, l)
            c_kv_weighted = attn_weights @ c_kv.view(batch_size, 1, -1, self.latent_dim)
            # Shape: (B, nh, N, E)
            out_per_head = c_kv_weighted @ self.W_absored
            out = out_per_head.sum(dim=1)

        return out

    def finish_training(self):
        # Isolate heads in W_out: (E, E) -> (nh, hd, E)
        W_out = self.out_proj.weight.T.view(
            self.num_heads, self.head_dim, self.embed_dim
        )

        # Isolate heads in W_uv: (l, E) -> (l, nh, hd) -> (nh, l, hd)
        W_uv = self.up_v_proj.weight.T.view(
            self.latent_dim, self.num_heads, self.head_dim
        ).transpose(0, 1)

        # Shape: (nh, l, E)
        self.W_absored = W_uv @ W_out

    def reset_cache(self) -> None:
        self.c_kv_cache.fill_(0)
        self.k_rope_cache.fill_(0)
        self.cache_len = 0
