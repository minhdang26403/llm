import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()

        t = torch.arange(0, embed_dim, 2)
        inv_freq = 10000 ** (-t / embed_dim)
        position = torch.arange(max_seq_len)
        angles = position.unsqueeze(1) * inv_freq  # Shape: (seq_len, d/2)

        # Optimized Interleaving: (seq_len, d/2, 2) -> (seq_len, d)
        # This is often faster than slice assignment [:, 0::2]
        embedding = torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(1)
        self.register_buffer("embedding", embedding)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.shape[1]
        return self.embedding[start_pos : start_pos + seq_len, :]


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len,
        base: int = 10_000,
        scaling_factor: float = 1,
        scaling_type: str = "",
    ):
        super().__init__()

        position = torch.arange(max_seq_len)  # Shape: (N,)

        if scaling_type == "ntk":
            base = base * scaling_factor ** (head_dim / (head_dim - 2))
        elif scaling_type == "linear":
            position /= scaling_factor

        t = torch.arange(0, head_dim, 2)
        inv_freq = base ** (-t / head_dim)  # Shape: (head_dim/2,)

        angles = torch.einsum("i,j->ij", position, inv_freq)  # Shape: (N, head_dim/2)
        angles = torch.cat([angles, angles], dim=-1)  # Shape: (N, head_dim)

        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x_left, x_right = x.chunk(2, dim=-1)
        return torch.cat([-x_right, x_left], dim=-1)

    # Inefficient!!!
    # def _rotate_adjacent(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Rotates adjacent pairs: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
    #     """
    #     x_rotated = torch.empty_like(x)
    #     # Evens take the negative of the odds
    #     x_rotated[..., 0::2] = -x[..., 1::2]
    #     # Odds take the evens
    #     x_rotated[..., 1::2] = x[..., 0::2]

    #     return x_rotated

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Apply RoPE to attention head states.

        Args:
            x: Input tensor with shape (batch_size, num_heads, seq_len, head_dim).

        Returns:
            Tensor with the same shape as `x`, after rotary transformation.

        Raises:
            ValueError: If the last dimension of `x` does not match configured `d`.
        """
        seq_len = x.shape[2]
        cos = self.cos_cached[start_pos : start_pos + seq_len, :].to(x.dtype)
        sin = self.sin_cached[start_pos : start_pos + seq_len, :].to(x.dtype)

        return cos * x + sin * self._rotate_half(x)
