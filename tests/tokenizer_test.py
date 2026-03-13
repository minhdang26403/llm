from pathlib import Path

import pytest

from tokenizer import Tokenizer


@pytest.fixture(scope="session")
def trained_tokenizer() -> Tokenizer:
    """Load tokenizer weights once per full pytest run."""
    repo_root = Path(__file__).resolve().parents[1]
    tokenizer_path = repo_root / "weights" / "bpe_tokenizer.json"
    if not tokenizer_path.exists():
        pytest.skip(
            f"Tokenizer weights not found at {tokenizer_path}. "
            "Run train_tokenizer.py first."
        )
    return Tokenizer.load(tokenizer_path)


def test_load_has_learned_merges(trained_tokenizer: Tokenizer) -> None:
    assert len(trained_tokenizer.merge_rules) > 0
    assert len(trained_tokenizer.vocab) > 256


def test_encode_decode_identity(trained_tokenizer: Tokenizer) -> None:
    text = (
        "On a rainy evening, the old library smelled of paper and dust, "
        "and every lamp along the aisle flickered softly."
    )
    token_ids = trained_tokenizer.encode(text)
    decoded = trained_tokenizer.decode(token_ids)
    assert decoded == text


def test_encode_is_deterministic(trained_tokenizer: Tokenizer) -> None:
    text = "Deterministic tokenization matters for reproducible training."
    token_ids_1 = trained_tokenizer.encode(text)
    token_ids_2 = trained_tokenizer.encode(text)
    assert token_ids_1 == token_ids_2


def test_unicode_roundtrip(trained_tokenizer: Tokenizer) -> None:
    text = "Xin chao! Emojis: 😀🚀 and accents: cafe, naive, jalapeno."
    token_ids = trained_tokenizer.encode(text)
    decoded = trained_tokenizer.decode(token_ids)
    assert decoded == text


def test_decode_invalid_token_raises(trained_tokenizer: Tokenizer) -> None:
    invalid_token_id = max(trained_tokenizer.unified_vocab.keys()) + 1
    with pytest.raises(ValueError, match="Invalid token id"):
        trained_tokenizer.decode([invalid_token_id])


def test_encode_empty_string(trained_tokenizer: Tokenizer) -> None:
    assert trained_tokenizer.encode("") == []
