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
        var = t.var(dim=self.normalized_dim, correction=0, keepdim=True)
        # Perform math on float32
        result = (t - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias

        return result.to(dtype)


class RMSNorm(nn.Module):
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
        self.scale = nn.Parameter(torch.ones(shape, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, dtype = x.float(), x.dtype
        inv_rms = torch.rsqrt(
            self.eps + torch.mean(t**2, dim=self.normalized_dim, keepdim=True)
        )
        # Perform math on float32
        result = t * inv_rms * self.scale

        # Need to convert back to the input's data type
        return result.to(dtype)
