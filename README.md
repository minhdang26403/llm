# Large Language Models from scratch

This project implements common Large Language Models from scratch.

## TODO
- Tokenizer
- Token Embedding
- Positional Embedding
- Transformer Block
  - LayerNorm 1
  - Causal Self-Attention
  - LayerNorm 2
  - MLP

## Project Layout

```
llm-from-scratch/
├── data/                  # Raw text files (e.g., tiny_shakespeare.txt)
├── weights/               # Saved .pt or .safetensors checkpoints
├── src/
│   ├── tokenizer.py       # Your BPE implementation
│   ├── dataset.py         # PyTorch Dataset & DataLoader
│   ├── models/            # Architecture definitions
│   │   ├── base.py        # Shared config (n_layer, n_head, d_model)
│   │   ├── gpt2.py        # GPT-2 implementation (MHA, Absolute Pos)
│   │   └── llama.py       # Llama implementation (GQA, RoPE, RMSNorm)
│   ├── layers/            # Reusable components
│   │   ├── attention.py   # MHA, GQA, MLA classes
│   │   ├── embeddings.py  # RoPE, Sinusoidal, Learned
│   │   └── norms.py       # LayerNorm vs RMSNorm
│   ├── train.py           # The Training Loop
│   └── generate.py        # The Inference/Sampling Engine
├── work_log.md            # Your notes on what worked/failed
└── config.yaml            # Hyperparameters (batch_size, lr, etc.)
```

## Dataset
TinyStories

## Unit Tests

```bash
pytest -q tests/attention_test.py
```

## Training Tokenizer
```bash
python src/train_tokenizer.py data/TinyStoriesV2-GPT4-train.txt --vocab-size 128256 --num-workers 12 --verbose
```

## Tokenizer Training Observations

Run setup:
- Dataset: TinyStoriesV2-GPT4-train (~2.1GB)
- Workers: 12
- Target vocab size: 128,256 (Llama 3 scale)
- Command:

```bash
python src/train_tokenizer.py data/TinyStoriesV2-GPT4-train.txt --vocab-size 128256 --num-workers 12 --verbose
```

Observed timing from one run:
- Stage 1 (segment boundaries): ~0.00s
- Stage 2 (pretokenization): ~34.07s
- Stage 3 (in-memory structures): ~0.11s
- Merge loop total: ~3.7s (from ~34.07s to ~37.78s elapsed)
- End-to-end tokenizer training: ~37.79s

Detailed output
```
Tokenizer stage [1/3] segmentation completed: segments=12, elapsed=0.00s
Tokenizer stage [2/3] pretokenization completed: workers=12, elapsed=34.07s
Tokenizer stage [3/3] in-memory structures completed: unique_words=59921, unique_pairs=2105, elapsed=0.11s
Starting tokenizer merge loop: target_vocab_size=128256, planned_merges=128000, progress_interval=6400
Tokenizer training progress: 6400/128000 merges (5.0%), current_vocab_size=6656, elapsed=36.72s
Tokenizer training progress: 12800/128000 merges (10.0%), current_vocab_size=13056, elapsed=37.02s
Tokenizer training progress: 19200/128000 merges (15.0%), current_vocab_size=19456, elapsed=37.22s
Tokenizer training progress: 25600/128000 merges (20.0%), current_vocab_size=25856, elapsed=37.35s
Tokenizer training progress: 32000/128000 merges (25.0%), current_vocab_size=32256, elapsed=37.45s
Tokenizer training progress: 38400/128000 merges (30.0%), current_vocab_size=38656, elapsed=37.52s
Tokenizer training progress: 44800/128000 merges (35.0%), current_vocab_size=45056, elapsed=37.59s
Tokenizer training progress: 51200/128000 merges (40.0%), current_vocab_size=51456, elapsed=37.65s
Tokenizer training progress: 57600/128000 merges (45.0%), current_vocab_size=57856, elapsed=37.70s
Tokenizer training progress: 64000/128000 merges (50.0%), current_vocab_size=64256, elapsed=37.75s
Tokenizer training progress: 70400/128000 merges (55.0%), current_vocab_size=70656, elapsed=37.78s
Tokenizer training progress: 72682/128000 merges (56.8%), current_vocab_size=72938, elapsed=37.78s
Tokenizer training completed: final_vocab_size=72938, elapsed=37.78s
Training time: 37.79s
```

Key takeaway:
- Pretokenization dominates total runtime.
- The merge process is very fast and scales well with larger target vocab sizes. In practice, increasing vocab targets (5k -> 10k -> 50k -> 100k -> 128k) showed only a small merge-time increase.


## Profile the code

```bash
python -m cProfile -o perf/prepare_train_dataset.prof -s tottime src/prepare_data.py --input data/TinyStoriesV2-GPT4-train.txt --output data/TinyStoriesV2-GPT4-train.bin --num-workers 12
```

```
python -c "import pstats; pstats.Stats('perf/prepare_train_dataset.prof').sort_stats('tottime').print_stats(20)" > perf/prepare_train_dataset_profile.txt
```
