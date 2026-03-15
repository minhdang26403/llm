# Large Language Models from scratch

The project aims to implement Large Language Models from scratch. In addition, it also implements training and inference infrastructure, including Byte-Pair Encoding (BPE) tokenizer , data pipeline, training loop, KV cache, sampling, and text generation.

## Project Layout

```text
llm-from-scratch/
├── data/                           # Raw text and pretokenized .bin datasets
├── perf/                           # Profiling artifacts and reports
├── src/
│   ├── tokenizer.py                # BPE tokenizer (train/save/load/encode/decode)
│   ├── train_tokenizer.py          # CLI to train and save tokenizer weights
│   ├── prepare_data.py             # Offline pretokenization pipeline (.txt -> .bin)
│   ├── prepare_data_workers.py     # Multiprocessing workers for pretokenization
│   ├── dataset.py                  # Memmap dataset over pretokenized token IDs
│   ├── train.py                    # Model training entrypoint (GPT/Llama)
│   ├── models/
│   │   ├── config.py               # Model config classes and defaults
│   │   ├── gpt.py                  # GPT model and blocks
│   │   └── llama.py                # Llama model and blocks
│   └── layers/                     # Reusable building blocks
│       ├── attention.py            # Causal attention, GQA/MQA support
│       ├── positional_embedding.py # Sinusoidal + RoPE
│       ├── norm.py                 # LayerNorm / RMSNorm
│       ├── activation.py           # SwiGLU
│       └── dropout.py              # Dropout layer
├── tests/                          # Unit tests
├── weights/                        # Saved tokenizer/model artifacts
└── README.md
```

## Dataset Introduction

The current development dataset is `TinyStoriesV2-GPT4`:
- `data/TinyStoriesV2-GPT4-train.txt`
- `data/TinyStoriesV2-GPT4-valid.txt`

Pipeline:
1. Train tokenizer on raw text.
2. Pretokenize raw text into a contiguous binary file of token IDs (`uint32`).
3. Train model by memory-mapping the binary file.

## Key Features

This project is interesting because:
- It covers both **ML modeling** and **systems engineering**.
- It is practical for experimentation with bottlenecks (CPU parsing, IPC, I/O, memory layout).
- It demonstrates how offline preprocessing can simplify and accelerate online training.

Implemented components:
- BPE tokenizer:
  - GPT-style pretokenization regex
  - special-token support
  - train/save/load/encode/decode
  - training observability with stage timings and progress
- Data preparation:
  - safe chunk boundary splitting
  - multiprocessing tokenization workers
  - shard merge to final `.bin` token file
- Dataset/data loading:
  - `numpy.memmap`-based token streaming
  - sequence slicing into `(inputs, targets)` pairs
- Models:
  - GPT and Llama variants
  - configurable attention/head layout
  - RoPE, RMSNorm, SwiGLU
- Training scaffolding:
  - CLI model selection
  - optimizer + scheduler + gradient clipping
  - optional validation and checkpointing
- Test suite:
  - attention, embedding, dropout, norm, tokenizer tests

## Run Unit Tests

Run all tests:

```bash
pytest -q tests
```

Run a single suite:

```bash
pytest -q tests/tokenizer_test.py
```

## Train Tokenizer

Example command:

```bash
python src/train_tokenizer.py data/TinyStoriesV2-GPT4-train.txt --vocab-size 128256 --num-workers 12 --verbose
```

Output:
- tokenizer weights saved to `weights/bpe_tokenizer.json`
- stage-wise timing + progress logs when `--verbose` is enabled

### Tokenizer Training Result (Observed)

Run setup:
- Dataset: TinyStoriesV2-GPT4-train (~2.1GB)
- Workers: 12
- Target vocab size: 128,256

Observed timing from one run:
- Stage 1 (segment boundaries): ~0.00s
- Stage 2 (pretokenization): ~34.07s
- Stage 3 (in-memory structures): ~0.11s
- Merge loop total: ~3.7s
- End-to-end tokenizer training: ~37.79s

Takeaway:
- Pretokenization dominates runtime.
- Merge loop is fast and scales very well as target vocab grows.

## Pretokenize Text to Binary Token IDs

Convert raw text to pretokenized `.bin`:

```bash
python src/prepare_data.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/TinyStoriesV2-GPT4-train.bin \
  --tokenizer weights/bpe_tokenizer.json \
  --chunk-size-bytes 16777216 \
  --num-workers 8
```

The output `.bin` stores contiguous `uint32` token IDs and is consumed by `src/dataset.py` via `numpy.memmap`.

## Profile the Data Pipeline

Collect profile:

```bash
python -m cProfile -o perf/prepare_train_dataset.prof src/prepare_data.py --input data/TinyStoriesV2-GPT4-train.txt --output data/TinyStoriesV2-GPT4-train.bin --num-workers 8
```

Generate top-time report:

```bash
python -c "import pstats; pstats.Stats('perf/prepare_train_dataset.prof').sort_stats('tottime').print_stats(30)" > perf/prepare_train_dataset_profile.txt
```

## Technical Deep-Dive
### Sequential vs Parallel Data Loading
For one-batch sanity checks on pretokenized `.bin` data:
- `num_workers=0` can be around `~0.01s`
- `num_workers=4` can be much slower (multi-second on macOS)

`DataLoader` workers parallelize `dataset.__getitem__`, but they add process/IPC overhead:
- worker process startup (spawn on macOS)
- inter-process queue handoff
- collation and synchronization costs

For this project, `dataset.__getitem__` is intentionally lightweight:
- `memmap` slice of contiguous integers
- small cast to tensor

Because the per-sample work is cheap, worker overhead can dominate first-batch latency.

Rule of thumb:
- Use `num_workers=0` (or `1`) for quick correctness checks and often for training too.
- Increase workers only when `__getitem__` is truly heavy (image decode/augment, spectrogram generation, on-the-fly tokenization, etc.).
- Benchmark in your exact environment before assuming more workers are better.
