import torch
import torch.nn as nn

from layers.norm import LayerNorm, RMSNorm


def test_layer_norm_single_dim():
    # NLP Example
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)

    layer_norm = LayerNorm(embedding_dim)
    output = layer_norm(embedding)

    torch_layer_norm = nn.LayerNorm(embedding_dim)
    expected = torch_layer_norm(embedding)

    assert torch.allclose(output, expected, atol=1e-6)


def test_layer_norm_multiple_dims():
    # Image Example
    N, C, H, W = 20, 5, 10, 10
    input = torch.randn(N, C, H, W)
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = LayerNorm([C, H, W])
    output = layer_norm(input)

    torch_layer_norm = nn.LayerNorm([C, H, W])
    expected = torch_layer_norm(input)

    assert torch.allclose(output, expected, atol=1e-6)


def test_rms_norm_single_dim():
    # NLP Example
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)

    layer_norm = RMSNorm(embedding_dim)
    output = layer_norm(embedding)

    torch_layer_norm = nn.RMSNorm(embedding_dim)
    expected = torch_layer_norm(embedding)

    assert torch.allclose(output, expected, atol=1e-6)


def test_rms_norm_multiple_dims():
    # Image Example
    N, C, H, W = 20, 5, 10, 10
    input = torch.randn(N, C, H, W)
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = RMSNorm([C, H, W])
    output = layer_norm(input)

    torch_layer_norm = nn.RMSNorm([C, H, W])
    expected = torch_layer_norm(input)

    assert torch.allclose(output, expected, atol=1e-6)
