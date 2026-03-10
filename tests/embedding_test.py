import torch

from layers.positional_embedding import SinusoidalEmbedding


def test_embedding_sanity():
    seq_len = 16
    max_seq_len = 128
    embed_dim = 64

    x = torch.randn(2, seq_len, embed_dim)

    embedding = SinusoidalEmbedding(embed_dim, max_seq_len)
    pe = embedding.forward(x)
    encoded_x = x + pe

    # Only check shape correctness
    assert encoded_x.shape == x.shape
