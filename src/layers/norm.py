import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(
        self, normalized_shape: int | list[int], eps: float = 1e-05, device=None
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            shape: tuple[int, ...] = (normalized_shape,)
        else:
            shape = tuple(normalized_shape)

        self.normalized_dim = tuple(range(-len(shape), 0))
        self.eps = eps

        # Initialize the tensor directly on the target hardware
        self.weight = nn.Parameter(
            torch.ones(shape, dtype=torch.float32, device=device)
        )
        self.bias = nn.Parameter(torch.zeros(shape, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, dtype = x.float(), x.dtype
        mean = t.mean(dim=self.normalized_dim, keepdim=True)
        diff = t - mean
        var = (diff**2).mean(dim=self.normalized_dim, keepdim=True)
        # Perform math on float32
        result = diff * torch.rsqrt(var + self.eps) * self.weight + self.bias

        return result.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int | list[int], eps=None, device=None):
        super().__init__()

        if isinstance(normalized_shape, int):
            shape: tuple[int, ...] = (normalized_shape,)
        else:
            shape = tuple(normalized_shape)

        self.normalized_dim = tuple(range(-len(shape), 0))
        self.eps = eps
        # Initialize the tensor directly on the target hardware
        self.scale = nn.Parameter(torch.ones(shape, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, dtype = x.float(), x.dtype
        eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        inv_rms = torch.rsqrt(
            eps + torch.mean(t**2, dim=self.normalized_dim, keepdim=True)
        )
        # Perform math on float32
        result = t * inv_rms * self.scale

        # Need to convert back to the input's data type
        return result.to(dtype)
