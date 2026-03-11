import torch

from layers.positional_embedding import RotaryPositionalEmbedding, SinusoidalEmbedding


def test_sinusoidal_embedding_sanity():
    seq_len = 16
    max_seq_len = 128
    embed_dim = 64

    x = torch.randn(2, seq_len, embed_dim)

    embedding = SinusoidalEmbedding(embed_dim, max_seq_len)
    pe = embedding.forward(x)
    encoded_x = x + pe

    # Only check shape correctness
    assert encoded_x.shape == x.shape


def test_sinusoidal_decay():
    embed_dim = 128
    max_seq_len = 100
    pe_layer = SinusoidalEmbedding(embed_dim, max_seq_len)

    # Get the raw embeddings for all positions
    # Shape: (max_seq_len, embed_dim)
    # Note: Access your internal buffer directly for the test
    embeddings = pe_layer.embedding

    # Measure similarity between position 0 and others
    pos0 = embeddings[0]
    similarities = [torch.dot(pos0, embeddings[i]).item() for i in range(max_seq_len)]

    # Logic: Position 0 should be most similar to Position 1,
    # and least similar to Position 99.
    assert similarities[1] > similarities[10]
    assert similarities[10] > similarities[99]


def test_rotary_embedding_sanity():
    batch_size = 2
    seq_len = 16
    max_seq_len = 128
    head_dim = 16
    num_heads = 4

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)

    embedding = RotaryPositionalEmbedding(head_dim, max_seq_len)
    rotated_q = embedding.forward(q)

    # Only check shape correctness
    assert rotated_q.shape == q.shape


def test_rope_relative_invariance():
    head_dim = 64
    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=100)

    # Create two random vectors (simulating a Query and a Key)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # Helper to rotate at a specific position
    def rotate_at_pos(t, pos):
        # We slice the cached cos/sin at the specific pos
        cos = rope.cos_cached[pos : pos + 1, :]
        sin = rope.sin_cached[pos : pos + 1, :]
        return (t * cos) + (rope._rotate_half(t) * sin)

    # Pair A: Positions 2 and 5 (delta = 3)
    q_2 = rotate_at_pos(q, 2)
    k_5 = rotate_at_pos(k, 5)
    dot_a = torch.sum(q_2 * k_5)

    # Pair B: Positions 10 and 13 (delta = 3)
    q_10 = rotate_at_pos(q, 10)
    k_13 = rotate_at_pos(k, 13)
    dot_b = torch.sum(q_10 * k_13)

    # They must be the same! (Allow for small float precision errors)
    assert torch.isclose(dot_a, dot_b, atol=1e-5)


def test_rope_magnitude_preservation():
    head_dim = 64
    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=100)
    x = torch.randn(1, 1, 1, head_dim)

    original_norm = torch.norm(x)
    rotated_x = rope.forward(x)
    rotated_norm = torch.norm(rotated_x)

    assert torch.isclose(original_norm, rotated_norm, atol=1e-6)
