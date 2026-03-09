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