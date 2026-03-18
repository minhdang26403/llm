import torch
import torch.nn.functional as F

from layers.attention import MultiheadAttention


def test_attention_tensor_shapes():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    # Standard MHA
    mha = MultiheadAttention(embed_dim=16, max_seq_len=8, num_heads=4, num_kv_heads=4)
    out_mha = mha(x)
    assert out_mha.shape == (2, 5, 16)

    # GQA
    gqa = MultiheadAttention(embed_dim=16, max_seq_len=8, num_heads=4, num_kv_heads=2)
    out_gqa = gqa(x)
    assert out_gqa.shape == (2, 5, 16)

    # MQA
    mqa = MultiheadAttention(embed_dim=16, max_seq_len=8, num_heads=4, num_kv_heads=1)
    out_mqa = mqa(x)
    assert out_mqa.shape == (2, 5, 16)


def test_attention_numerical_correctness():
    embed_dim = 64
    max_seq_len = 128
    num_heads = 4
    num_kv_heads = 2
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Our MHA implementation
    attn = MultiheadAttention(embed_dim, max_seq_len, num_heads, num_kv_heads)

    # PyTorch implementation
    with torch.no_grad():
        # Compute attention using our implementation
        out = attn(x)

        head_dim = embed_dim // num_heads
        query = (
            attn.q_proj(x)
            .view(batch_size, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            attn.k_proj(x)
            .view(batch_size, seq_len, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        value = (
            attn.v_proj(x)
            .view(batch_size, seq_len, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        expected = (
            F.scaled_dot_product_attention(
                query, key, value, is_causal=True, enable_gqa=True
            )
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )
        expected = attn.out_proj(expected)

    assert torch.allclose(out, expected, atol=1e-6)


def test_attention_causality_no_future_leakage():
    torch.manual_seed(123)
    attn = MultiheadAttention(embed_dim=8, max_seq_len=8, num_heads=2, num_kv_heads=2)

    x_base = torch.randn(1, 6, 8)
    x_changed = x_base.clone()
    x_changed[:, 3:, :] = torch.randn_like(x_changed[:, 3:, :]) * 50.0

    out_base = attn(x_base)
    out_changed = attn(x_changed)

    # Positions [0, 1, 2] should be unaffected by modifications to positions >= 3.
    assert torch.allclose(out_base[:, :3, :], out_changed[:, :3, :], atol=1e-6)


def test_attention_with_kv_cache():
    torch.manual_seed(0)
    embed_dim = 64
    max_seq_len = 128
    num_heads = 4
    num_kv_heads = 2
    batch_size = 1
    seq_len = 16
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Our MHA implementation
    attn = MultiheadAttention(
        embed_dim, max_seq_len, num_heads, num_kv_heads, use_cache=True
    )

    # PyTorch implementation
    with torch.no_grad():
        # Compute attention using our implementation
        out = attn(x)
        assert attn.cache_len == seq_len

        head_dim = embed_dim // num_heads
        key = (
            attn.k_proj(x)
            .view(batch_size, seq_len, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        value = (
            attn.v_proj(x)
            .view(batch_size, seq_len, num_kv_heads, head_dim)
            .transpose(1, 2)
        )

    k_cache = attn.k_cache
    v_cache = attn.v_cache
    assert torch.equal(k_cache[:, :, :seq_len, :], key)
    assert torch.equal(v_cache[:, :, :seq_len, :], value)
    assert attn.cache_len == seq_len

    # Reset cache
    attn.reset_cache()
    assert torch.equal(k_cache, torch.zeros_like(k_cache))
    assert torch.equal(v_cache, torch.zeros_like(v_cache))
    assert attn.cache_len == 0


def test_attention_with_kv_cache_numerical_correctness():
    torch.manual_seed(0)
    embed_dim = 8
    max_seq_len = 128
    num_heads = 4
    num_kv_heads = 2
    batch_size = 1
    seq_len = 16
    prefill_length = 10
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Our MHA implementation (no cache)
    attn = MultiheadAttention(embed_dim, max_seq_len, num_heads, num_kv_heads)

    # Cached attention with identical weights.
    attn_cache = MultiheadAttention(
        embed_dim, max_seq_len, num_heads, num_kv_heads, use_cache=True
    )
    attn_cache.load_state_dict(attn.state_dict(), strict=True)

    with torch.no_grad():
        out = attn(x[:, : prefill_length + 1, :])
        _ = attn_cache(x[:, :prefill_length, :])
        decode_out = attn_cache(x[:, prefill_length : prefill_length + 1, :])

    assert torch.allclose(out[:, -1, :], decode_out)
