# Large Language Models from Scratch: An ML Infrastructure Approach
This project is a complete, end-to-end implementation of Large Language Models (LLMs) built entirely from the ground up. It goes beyond standard model architecture by placing a heavy emphasis on **ML systems engineering and infrastructure**.

Rather than relying on pre-built abstractions, this repository implements the critical backend systems required for efficient training and inference. It serves as a practical sandbox for tackling real-world computational bottlenecks—such as CPU parsing limits, Inter-Process Communication (IPC) overhead, concurrent file I/O, and memory layout optimization.

Core infrastructure built from scratch includes:
- A custom, highly optimized Byte-Pair Encoding (BPE) tokenizer.
- A Map-Reduce multiprocessing data pipeline for blazing-fast, concurrent pre-tokenization.
- Zero-copy, `numpy.memmap`-based data streaming for $O(1)$ training batch loading.
- Full training loops and inference engines featuring KV caching, configurable sampling, and text generation.

## ⚙️ Installation
This project uses standard Python virtual environments and is configured via `pyproject.toml`.

Clone the repository and run the following commands to set up an isolated environment:
```bash
# 1. Create a clean virtual environment
python3 -m venv .venv

# 2. Activate the environment
source .venv/bin/activate

# 3. Upgrade pip to the latest version
python -m pip install --upgrade pip

# 4. Install the package in editable mode along with development dependencies
python -m pip install -e ".[dev]"
```

## 📂 Project Architecture
The codebase is strictly separated into modular components, isolating the offline data pipeline from the online training loops.

```text
llm/
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

## 📊 Dataset & Pre-processing Pipeline
The primary development dataset for this project is `TinyStoriesV2-GPT4`. The pipeline requires raw text to be split into training and validation sets:
- `data/TinyStoriesV2-GPT4-train.txt`
- `data/TinyStoriesV2-GPT4-valid.txt`

To maximize GPU utilization during online training, all text parsing, regex operations, and Byte-Pair Encoding (BPE) are pushed to an offline pre-processing pipeline.

### Step 1: Train the Tokenizer
The custom BPE tokenizer implements GPT-style pre-tokenization regex and supports special tokens.

Train the tokenizer on the raw text using the following command:
```bash
python src/train_tokenizer.py data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 128256 \
  --num-workers 12 \
  --verbose
```

**Observed Performance:** Training a tokenizer from scratch on a massive corpus is notoriously CPU-bound. By implementing a multiprocessing architecture, the tokenizer scales exceptionally well.

- **Hardware:** Apple Silicon Mac (12 workers)
- **Dataset:** TinyStoriesV2-GPT4-train (~2.1GB)
- **Target Vocab Size:** 128,256
- **Stage 1 (Segment boundaries):** ~0.00s
- **Stage 2 (Pre-tokenization):** ~34.07s
- **Stage 3 (In-memory structures):** ~0.11s
- **Merge loop total:** ~3.7s
- **End-to-end tokenizer training:** ~37.79s

The tokenizer weights and vocabulary are automatically saved to `weights/bpe_tokenizer.json`.

### Step 2: Pre-tokenize the Text to Binary Shards
Once the tokenizer is trained, the raw `.txt` files must be converted into contiguous binary files (`.bin`) of token IDs. This script utilizes a highly optimized Map-Reduce architecture. It splits the dataset into safe chunk boundaries, processes them concurrently across isolated worker processes, and merges the resulting shards.

Run the pre-tokenization script for the **training set**:

```bash
python src/prepare_data.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/TinyStoriesV2-GPT4-train.bin \
  --tokenizer weights/bpe_tokenizer.json \
  --chunk-size-bytes 16777216 \
  --num-workers 8
```

Run the pre-tokenization script for the **validation set**:

```bash
python src/prepare_data.py \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/TinyStoriesV2-GPT4-valid.bin \
  --tokenizer weights/bpe_tokenizer.json \
  --chunk-size-bytes 16777216 \
  --num-workers 8
```

**Observed Performance:** Thanks to independent worker caching and the elimination of Python's Global Interpreter Lock (GIL) via `multiprocessing.Pool`, the pipeline processes the entire 2.1GB training set (over 539,000,000 tokens) and writes the highly compressed `uint32` binary file in **~22.7 seconds**. Before optimization, this pre-tokenization step took **~253 seconds**.

## 🚀 Model Training
With the BPE tokenizer trained and the raw text successfully compiled into highly optimized `.bin` shards, the model is ready to train.

The training loop (`src/train.py`) acts as the primary entry point. Because the `TextDataset` relies on zero-copy `numpy.memmap` to stream token IDs directly from the SSD, the data loading overhead is effectively $O(1)$, easily saturating the GPU without needing complex multiprocessing loaders.

### Launching the Training Loop
The training scaffolding provides CLI arguments to select the model architecture, hyperparameters, and checkpointing intervals. You can train either a standard GPT-style architecture or a Llama-style architecture (featuring RMSNorm, SwiGLU, and RoPE).

Train a GPT variant:
```bash
python src/train.py gpt \
  data/TinyStoriesV2-GPT4-train.bin \
  --val-dataset-path data/TinyStoriesV2-GPT4-valid.bin \
  --num-workers 0 \
  --batch-size 8 \
  --num-epochs 4
```

Train a Llama variant:
```bash
python src/train.py llama \
  data/TinyStoriesV2-GPT4-train.bin \
  --val-dataset-path data/TinyStoriesV2-GPT4-valid.bin \
  --num-workers 0 \
  --batch-size 8 \
  --num-epochs 4
```

### Validation & Checkpointing
The training loop includes built-in scaffolding for periodic validation against the validation dataset and automatic checkpointing.

During training, model checkpoints are saved to the `checkpoints/` directory by default (or a custom path via `--output-dir`). These artifacts can be loaded later to resume training from an exact step or to perform offline text generation.

## 🔮 Inference & Generation (Work in Progress)
While the offline data pipeline and online training loops are fully operational, the inference engine is currently under active development.

The goal of this module is to efficiently load the trained checkpoints and serve the model for interactive text generation. Future updates to this repository will include a from-scratch implementation of the inference infrastructure, specifically focusing on:

- **KV Caching:** Implementing key-value caches to prevent redundant attention matrix computations during autoregressive decoding.
- **Sampling Strategies:** Adding configurable temperature, top-k, and top-p sampling for high-quality text generation.
- **Interactive CLI:** A command-line interface to feed prompts into the loaded GPT or Llama variants and stream the generated text back to the user.

## ⏱️ Profiling & Performance
Because this project is heavily focused on systems optimization, profiling tools are used extensively to identify and eliminate CPU, I/O, and memory bottlenecks.

You can profile the offline pre-tokenization pipeline using Python's built-in `cProfile` to see exactly where the interpreter spends its time:
```bash
python -m cProfile -o perf/prepare_train_dataset.prof \
  src/prepare_data.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/TinyStoriesV2-GPT4-train.bin \
  --num-workers 8
```

Generate a top-time report from the profiling artifacts:
```bash
python -c "import pstats; pstats.Stats('perf/prepare_train_dataset.prof').sort_stats('tottime').print_stats(30)" > perf/prepare_train_dataset_profile.txt
```

### Analyzing the Results
When running the pre-tokenization pipeline on an Apple Silicon Mac with 8 physical workers, the pipeline processed **539,309,852 tokens in ~22.77 seconds**.

This represents an 11x speedup over a naive sequential implementation, achieving super-linear scaling. The `cProfile` analysis reveals exactly how this was accomplished:

1. **Main Process Offloading:** The profile shows the main process spends almost its entire runtime sitting on `posix.read`. Because we use the `spawn` start method and a Map-Reduce architecture, the main process does zero encoding work. It simply orchestrates the task queue and waits for the Inter-Process Communication (IPC) pipes to return the paths of the completed binary shards.
2. **Dynamic Load Balancing:** By breaking the 2GB dataset into 213 smaller chunks (instead of 8 massive chunks), the multiprocessing pool continuously feeds the workers. If one worker hits a complex, regex-heavy text segment, the other 7 cores don't stall. They keep pulling from the queue, maintaining 100% CPU utilization across all cores and entirely eliminating the "straggler effect."
3. **I/O Bound Amortization:** Shard merging using `shutil.copyfileobj` with a 16MB buffer takes less than 1 second of the total runtime, proving that disk write speeds are no longer the bottleneck once the text is converted to a highly compressed `uint32` format.

## 🔬 Technical Deep-Dives
This project is designed as a practical experimentation ground for understanding system bottlenecks—such as CPU parsing, Inter-Process Communication (IPC), I/O throughput, and memory layout—when training Large Language Models. Below are the core engineering decisions made to optimize the pipeline.

### 1. Map-Reduce Pre-tokenization & Dynamic Load Balancing
**The Problem:** Tokenizing gigabytes of text using Byte-Pair Encoding (BPE) regex is heavily CPU-bound. Standard Python execution is restricted to a single core by the Global Interpreter Lock (GIL). Furthermore, naively splitting a 2GB file into 8 massive chunks for 8 workers causes severe memory bloat and introduces the "straggler effect," where fast workers finish early and sit idle while one worker struggles with a complex text chunk.

**The Solution:** The pipeline utilizes a Map-Reduce architecture with **Dynamic Load Balancing**.

- Instead of massive chunks, the dataset is logically divided into hundreds of smaller, boundary-safe segments (e.g., 16MB each).
- A `multiprocessing.Pool` maintains a queue of these tasks. As workers finish a chunk, they instantly pull the next one.
- This guarantees 100% CPU utilization across all cores for the entire duration of the run.
- To avoid the IPC overhead of pickling the massive BPE vocabulary dictionary hundreds of times, the `multiprocessing` initializer pattern is used to load the tokenizer into each worker's isolated global memory exactly once at boot up.

### 2. Zero-Copy Memory Mapping (`numpy.memmap`)
**The Problem:** Online text parsing during the training loop is a massive anti-pattern. Reading raw strings, running regex, and allocating Python lists of integers on the fly creates severe CPU bottlenecks that leave the GPU starved for data.

**The Solution:** All text processing is pushed offline, generating a single, contiguous `.bin` file of `uint32` token IDs.

- The PyTorch `Dataset` uses `numpy.memmap` to map this binary file directly into virtual memory.
- Unlike standard Python list slicing (which creates a physical copy of the data in RAM, an $O(N)$ operation), `memmap` allows PyTorch to take a **zero-copy view** of the underlying byte stream.
- Slicing a context window of 1024 tokens becomes an $O(1)$ pointer operation, completely bypassing the Python interpreter's memory overhead and keeping system RAM usage flat, regardless of whether the dataset is 2GB or 2TB.

### 3. Sequential vs Parallel Data Loading
A common misconception in PyTorch is that increasing `DataLoader` workers always increases speed. However, for one-batch sanity checks on pretokenized `.bin` data, `num_workers=0` can complete in around `~0.01s`, whereas `num_workers=4` can be much slower (taking multiple seconds on macOS).

**The Architecture:** `DataLoader` workers parallelize `dataset.__getitem__`, but they introduce significant process and IPC overhead. This includes:

- Worker process startup time (which relies on the highly expensive `spawn` method on macOS).
- Inter-process queue handoff.
- Collation and synchronization costs.

**The Decision:** For this project, `dataset.__getitem__` is intentionally lightweight, consisting only of a `memmap` slice of contiguous integers and a small cast to a tensor. Because the per-sample work is so cheap, the worker overhead easily dominates the first-batch latency.

**Rule of Thumb:**

- Use `num_workers=0` (or `1`) for quick correctness checks, and often for full training runs over `memmap` datasets.
- Only increase workers when `__getitem__` is truly heavy (e.g., image decoding/augmentation, spectrogram generation, or on-the-fly tokenization).
- Always benchmark in the exact execution environment before assuming more workers yield better performance.

## 🧪 Running Tests
To ensure the cor-ectness of the custom infrastructure and model architecture, this repository includes a comprehensive unit testing suite.

The tests validate the exact mathematical behavior and tensor shapes of the implemented components, including:
- Custom BPE Tokenizer encoding/decoding loops.
- Causal Attention mechanisms (including GQA/MQA support).
- Positional Embeddings (Sinusoidal and RoPE).
- Custom Normalization (LayerNorm and RMSNorm).
- Dropout layers and SwiGLU activations.

Run the entire test suite:
```bash
pytest -q tests
```

Run a specific test suite (e.g., tokenizer validation):
```bash
pytest -q tests/tokenizer_test.py
```

## Distributed Training
```bash
GLOO_SOCKET_IFNAME=lo0 GLOO_DISABLE_IPV6=1 torchrun --nproc_per_node=2 apps/ddp_train.py
```

## 🗺️ Future Work
While the core data pipeline and training scaffolding are complete, the next phase of this project will focus heavily on scaling, distributed systems, and hardware-level optimizations.

Planned features and infrastructure upgrades include:
- **High-Performance Attention:** Integrating hardware-aware algorithms like FlashAttention to minimize memory bandwidth bottlenecks (HBM reads/writes) during the attention computation.
- **Distributed Training Infrastructure:** Scaling the training loop beyond a single device by implementing distributed training techniques, including Data Parallelism, Tensor Parallelism, Pipeline Parallelism, and ZeRO (Zero Redundancy Optimizer) memory savings.
- **Advanced Inference & Serving:** Evolving the text generation script into a robust serving system. This includes implementing continuous batching for high-throughput requests and advanced KV cache management (such as PagedAttention) to optimize memory layout during generation.
- **Custom GPU Kernels:** Replacing high-level PyTorch operations with custom-written CUDA kernels to push hardware limits, maximize GPU occupancy, and explicitly map operations against the Roofline model.
