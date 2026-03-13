import torch
import torch.nn as nn

from layers.activation import SwiGLU
from layers.attention import MultiheadAttention
from layers.norm import RMSNorm
from layers.positional_embedding import RotaryPositionalEmbedding
from models.config import LlamaConfig


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig, rope: RotaryPositionalEmbedding):
        super().__init__()

        self.rms_norm_1 = RMSNorm(config.embed_dim, eps=1e-5)
        self.rope = RotaryPositionalEmbedding(
            config.embed_dim // config.num_heads,
            config.max_seq_len,
            scaling_factor=config.scaling_factor,
            scaling_type=config.scaling_type,
        )
        self.attention = MultiheadAttention(
            config.embed_dim,
            config.max_seq_len,
            config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout_rate=config.attn_pdrop,
            rope=rope,
        )
        self.rms_norm_2 = RMSNorm(config.embed_dim, eps=1e-5)
        self.swiglu_ffn = SwiGLU(config.embed_dim, config.ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.rms_norm_1(x))
        x = x + self.swiglu_ffn(self.rms_norm_2(x))
        return x


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        rope = RotaryPositionalEmbedding(
            config.embed_dim // config.num_heads,
            config.max_seq_len,
            scaling_factor=config.scaling_factor,
            scaling_type=config.scaling_type,
        )
        self.llama_blocks = nn.ModuleList(
            LlamaBlock(config, rope) for _ in range(config.num_blocks)
        )
        self.rms_norm = RMSNorm(config.embed_dim, eps=1e-5)
        # Llama models do not use weight tying
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        for block in self.llama_blocks:
            x = block(x)

        x = self.rms_norm(x)
        logits = self.lm_head(x)

        return logits
