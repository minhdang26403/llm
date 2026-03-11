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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return self.embedding[:seq_len, :]


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d,
        max_seq_len,
        base: int = 10_000,
        scaling_factor: float = 1,
        scaling_type: str = "",
    ):
        super().__init__()

        position = torch.arange(max_seq_len)  # Shape: (N,)

        if scaling_type == "ntk":
            base = base * scaling_factor ** (d / (d - 2))
        elif scaling_type == "linear":
            position /= scaling_factor

        t = torch.arange(0, d, 2)
        inv_freq = base ** (-t / d)  # Shape: (d/2,)

        angles = torch.einsum("i,j->ij", position, inv_freq)  # Shape: (N, d/2)
        angles = torch.cat([angles, angles], dim=-1)  # Shape: (N, d)

        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x_left, x_right = x.chunk(2, dim=-1)
        return torch.cat([-x_right, x_left], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]

        return cos * x + sin * self._rotate_half(x)
