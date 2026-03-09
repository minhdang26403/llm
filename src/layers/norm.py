import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | list[int], eps=1e-05):
        super().__init__()  # CRITICAL: Always call super()

        if isinstance(normalized_shape, int):
            shape = (normalized_shape,)
        else:
            shape = tuple(normalized_shape)  # type: ignore

        self.normalized_dim = tuple(range(-len(shape), 0))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=self.normalized_dim, keepdim=True)
        var = x.var(dim=self.normalized_dim, correction=0, keepdim=True)

        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int | list[int], eps=None):
        super().__init__()

        if isinstance(normalized_shape, int):
            shape = (normalized_shape,)
        else:
            shape = tuple(normalized_shape)  # type: ignore

        self.normalized_dim = tuple(range(-len(shape), 0))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps if self.eps else torch.finfo(x.dtype).eps
        rms = torch.sqrt(eps + torch.mean(x**2, dim=self.normalized_dim, keepdim=True))

        return x / rms * self.weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
