"""CLI utility to train and save the BPE tokenizer."""

import argparse
from pathlib import Path
import time

from tokenizer import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "weights" / "bpe_tokenizer.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer on a dataset file and save checkpoint."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to dataset text file used for tokenizer training.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32768,
        help="Target tokenizer vocab size (default: 32768).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes used during pretokenization (default: 4).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print periodic tokenizer training progress.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not dataset_path.is_file():
        raise ValueError(f"Dataset path must be a file: {dataset_path}")

    tokenizer = Tokenizer(
        file_path=dataset_path,
        vocab_size=args.vocab_size,
        num_workers=args.num_workers,
    )
    train_start = time.perf_counter()
    tokenizer.train(verbose=args.verbose)
    train_seconds = time.perf_counter() - train_start
    tokenizer.save(DEFAULT_OUTPUT_PATH)

    print(f"Tokenizer trained from: {dataset_path}")
    print(f"Training time: {train_seconds:.2f}s")
    print(f"Tokenizer weights saved to: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
