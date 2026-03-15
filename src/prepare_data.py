import argparse
import multiprocessing
import shutil
import time
from pathlib import Path

from prepare_data_workers import encode_worker, init_worker
from tokenizer import Tokenizer, get_worker_segment_boundaries


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

    # Load just to get special tokens for boundary calculation
    print("Loading tokenizer in main process for boundary calculation...")
    tokenizer = Tokenizer.load(tokenizer_path)

    file_size = input_path.stat().st_size
    num_chunks = (file_size + chunk_size_bytes - 1) // chunk_size_bytes

    print(f"Calculating safe boundaries for {num_chunks} chunks...")
    boundaries = get_worker_segment_boundaries(
        input_path, tokenizer.special_tokens, num_chunks
    )

    args = [
        (chunk_idx, start, end, input_path, output_path)
        for chunk_idx, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Spawning {num_workers} workers...")
    with multiprocessing.Pool(
        processes=num_workers, initializer=init_worker, initargs=(tokenizer_path,)
    ) as p:
        results = p.map(encode_worker, args)

    # Separate the paths and the token counts returned by the workers
    output_shards = [res[0] for res in results]
    total_tokens = sum(res[1] for res in results)

    print(f"Merging {len(output_shards)} shards into {output_path}...")
    merge_shards(output_shards, output_path)

    process_time_secs = time.perf_counter() - start_time
    print(f"Done! Processed {total_tokens:,} tokens in {process_time_secs:.2f}s")


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
        default=16 * 1024 * 1024,
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
        args.input,
        args.output,
        args.tokenizer,
        args.chunk_size_bytes,
        args.num_workers,
    )
