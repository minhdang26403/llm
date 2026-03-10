import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()

        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        mask = torch.rand_like(x) > self.p
        if self.inplace:
            x *= mask
            x /= 1 - self.p
        else:
            x = x * mask / (1 - self.p)

        return x
