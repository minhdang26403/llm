from pathlib import Path

import numpy as np

from tokenizer import Tokenizer

# Each worker process has its own isolated module globals.
_GLOBAL_TOKENIZER: Tokenizer | None = None


def init_worker(tokenizer_path: Path) -> None:
    """Initialize per-process tokenizer once for worker reuse."""
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = Tokenizer.load(tokenizer_path)


def encode_worker(args: tuple[int, int, int, Path, Path]) -> tuple[Path, int]:
    """Encode one byte-range chunk and write it to a temporary shard file."""
    chunk_idx, start, end, input_path, output_path = args

    global _GLOBAL_TOKENIZER
    assert _GLOBAL_TOKENIZER is not None

    output_path_private = output_path.with_stem(f"{output_path.stem}_{chunk_idx}")
    with open(output_path_private, "wb") as out_f:
        with open(input_path, "rb") as in_f:
            in_f.seek(start)
            chunk_bytes = in_f.read(end - start)
            chunk_text = chunk_bytes.decode("utf-8")

            token_ids = _GLOBAL_TOKENIZER.encode(chunk_text)
            np_array = np.array(token_ids, dtype=np.uint32)
            out_f.write(np_array.tobytes())

    return output_path_private, len(token_ids)
