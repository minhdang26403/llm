import torch
import torch.distributed as dist
import torch.nn as nn

from distributed.tensor_parallel.layers import ParallelAttention, ParallelSwiGLU
from layers.norm import RMSNorm
from layers.positional_embedding import RotaryPositionalEmbedding
from models.config import LlamaConfig


def _require_distributed_initialized() -> None:
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed process group must be initialized before "
            "constructing ParallelLlama modules."
        )


class ParallelLlamaBlock(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        rope: RotaryPositionalEmbedding,
        tp_group: dist.ProcessGroup | None = None,
    ):
        super().__init__()
        _require_distributed_initialized()

        self.rms_norm_1 = RMSNorm(config.embed_dim, eps=1e-5)
        self.attention = ParallelAttention(
            max_seq_len=config.max_seq_len,
            embed_dim=config.embed_dim,
            head_dim=config.head_dim,
            tp_group=tp_group,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout_rate=config.attn_pdrop,
            rope=rope,
            use_cache=config.use_cache,
        )
        self.rms_norm_2 = RMSNorm(config.embed_dim, eps=1e-5)
        self.swiglu_ffn = ParallelSwiGLU(config.embed_dim, config.ffn_dim, tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.rms_norm_1(x))
        x = x + self.swiglu_ffn(self.rms_norm_2(x))
        return x

    def reset_cache(self) -> None:
        self.attention.reset_cache()


class ParallelLlama(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        tp_group: dist.ProcessGroup | None = None,
    ):
        super().__init__()
        _require_distributed_initialized()

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        rope = RotaryPositionalEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            scaling_factor=config.scaling_factor,
            scaling_type=config.scaling_type,
        )
        self.llama_blocks = nn.ModuleList(
            ParallelLlamaBlock(config, rope, tp_group=tp_group)
            for _ in range(config.num_blocks)
        )
        self.rms_norm = RMSNorm(config.embed_dim, eps=1e-5)
        # Keep output projection replicated; tensor parallelism is applied inside blocks
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Performs the forward pass in the tensor-parallel Llama model.
        Parameters:
            x: input token ids
            start_pos: retained for parity with Llama.forward() signature.
        """
        _ = start_pos
        x = self.token_embedding(x)
        for block in self.llama_blocks:
            x = block(x)

        x = self.rms_norm(x)
        logits = self.lm_head(x)
        return logits

    def reset_cache(self) -> None:
        for block in self.llama_blocks:
            assert isinstance(block, ParallelLlamaBlock)
            block.reset_cache()
