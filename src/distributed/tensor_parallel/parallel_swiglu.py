import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .parallel_linear import ColumnParallelLinear, RowParallelLinear


class ParallelSwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, tp_group: dist.ProcessGroup):
        super().__init__()

        self.gate_proj = ColumnParallelLinear(d_model, d_hidden, tp_group)
        self.value_proj = ColumnParallelLinear(d_model, d_hidden, tp_group)
        self.out_proj = RowParallelLinear(d_hidden, d_model, tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        value = self.value_proj(x)
        hidden = gate * value
        return self.out_proj(hidden)
