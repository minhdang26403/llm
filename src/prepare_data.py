import argparse
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tokenizer import Tokenizer, get_worker_segment_boundaries


def _encode_worker(args: tuple[int, int, int, Path, Path, Path]) -> Path:
    chunk_idx, start, end, input_path, output_path, tokenizer_path = args

    # Each worker loads its own tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.load(tokenizer_path)

    # Create a private file to write encoded tokens into
    output_path_private = output_path.with_stem(f"{output_path.stem}_{chunk_idx}")

    with open(output_path_private, "wb") as out_f:
        with open(input_path, "rb") as in_f:
            # Read just this safe chunk into memory
            in_f.seek(start)
            chunk_bytes = in_f.read(end - start)
            chunk_text = chunk_bytes.decode("utf-8")

            # Encode the text into Python integers
            token_ids = tokenizer.encode(chunk_text)

            # Convert to a highly compressed numpy array and write to disk
            # uint32 should be enough to handle all known vocab size
            np_array = np.array(token_ids, dtype=np.uint32)
            out_f.write(np_array.tobytes())

    return output_path_private


def merge_shards(output_shards: list[Path], final_output: Path) -> None:
    with open(final_output, "wb") as out_f:
        for shard in output_shards:
            with open(shard, "rb") as in_f:
                # Use a 16MB buffer to maximize disk throughput
                shutil.copyfileobj(in_f, out_f, 16 * 1024 * 1024)
            # Clean up the temporary file immediately after copying
            shard.unlink()

    print(f"Binary file saved at: {final_output}")


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    tokenizer_path: Path,
    chunk_size_bytes: int,
    num_workers: int,
):
    start_time = time.perf_counter()
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.load(tokenizer_path)

    file_size = input_path.stat().st_size
    num_chunks = (file_size + chunk_size_bytes - 1) // chunk_size_bytes
    print(f"Calculating safe boundaries for {num_chunks} chunks...")
    boundaries = get_worker_segment_boundaries(
        input_path, tokenizer.special_tokens, num_chunks
    )

    args = [
        (chunk_idx, start, end, input_path, output_path, tokenizer_path)
        for chunk_idx, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
    ]

    with Pool(num_workers) as p:
        output_shards = p.map(_encode_worker, args)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Pre-tokenizing and saving to {output_path}...")
    merge_shards(output_shards, output_path)

    process_time_secs = time.perf_counter() - start_time
    print(f"Done! Processed the entire file in {process_time_secs}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Raw text file")
    parser.add_argument("--output", type=Path, required=True, help="Output .bin file")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("weights/bpe_tokenizer.json"),
        help="Tokenizer JSON",
    )
    parser.add_argument(
        "--chunk-size-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="Approximate bytes per chunk before tokenization.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes used during pretokenization (default: 4).",
    )
    args = parser.parse_args()

    prepare_dataset(
        args.input_path,
        args.output_path,
        args.tokenizer_path,
        args.chunk_size_bytes,
        args.num_workers,
    )
