import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.out_proj = nn.Linear(d_hidden, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        value = self.value_proj(x)
        out = self.out_proj(gate * value)
        return out
