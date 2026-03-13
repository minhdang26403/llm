from dataclasses import dataclass
from typing import Literal


RopeScalingType = Literal["", "linear", "ntk"]


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    embed_dim: int
    num_heads: int
    num_kv_heads: int
    num_blocks: int
    ffn_dim: int
    dropout_rate: float = 0.0
    attn_pdrop: float = 0.0
    scaling_factor: float = 1.0
    scaling_type: RopeScalingType = ""

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")
        if not 0.0 <= self.attn_pdrop <= 1.0:
            raise ValueError("attn_pdrop must be in [0, 1]")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive")


@dataclass
class GPTConfig(ModelConfig):
    @classmethod
    def default(cls) -> "GPTConfig":
        # GPT-like small baseline for local experiments.
        return cls(
            vocab_size=128_256,
            max_seq_len=1024,
            embed_dim=768,
            num_heads=12,
            num_kv_heads=12,
            num_blocks=12,
            ffn_dim=3072,
            dropout_rate=0.1,
            attn_pdrop=0.1,
            scaling_factor=1.0,
            scaling_type="",
        )


@dataclass
class LlamaConfig(ModelConfig):
    @classmethod
    def default(cls) -> "LlamaConfig":
        # Llama-style small-ish baseline (GQA enabled).
        return cls(
            vocab_size=128_256,
            max_seq_len=2048,
            embed_dim=1024,
            num_heads=16,
            num_kv_heads=8,
            num_blocks=16,
            ffn_dim=2816,
            dropout_rate=0.0,
            attn_pdrop=0.0,
            scaling_factor=1.0,
            scaling_type="",
        )
