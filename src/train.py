"""Basic training entrypoint with model/config selection from CLI."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from dataset import TextDataset
from models.config import GPTConfig, LlamaConfig, ModelConfig
from models.gpt import GPT
from models.llama import Llama
from tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a large language model.")
    parser.add_argument("model_name", choices=["gpt", "llama"], help="Model family.")
    parser.add_argument("dataset_path", type=Path, help="Training dataset text file.")
    parser.add_argument(
        "--val-dataset-path",
        type=Path,
        default=None,
        help="Optional validation dataset text file.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("weights/bpe_tokenizer.json"),
        help="Tokenizer checkpoint path.",
    )
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers."
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Base LR.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW decay.")
    parser.add_argument(
        "--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm."
    )
    parser.add_argument("--log-every", type=int, default=10, help="Train log interval.")
    parser.add_argument(
        "--val-every",
        type=int,
        default=500,
        help="Run validation every N train steps (if val loader exists).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save checkpoints.",
    )
    return parser.parse_args()


def validate_training_args(args: argparse.Namespace) -> None:
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    if args.val_every <= 0:
        raise ValueError("--val-every must be > 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    if not tokenizer_path.exists():
        raise RuntimeError(
            f"Tokenizer weights not found at {tokenizer_path}. "
            "Run train_tokenizer.py first."
        )
    return Tokenizer.load(tokenizer_path)


def create_model_and_config(model_name: str) -> tuple[nn.Module, ModelConfig]:
    if model_name == "gpt":
        config = GPTConfig.default()
        return GPT(config), config
    elif model_name == "llama":
        config = LlamaConfig.default()
        return Llama(config), config
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def get_scheduler(optimizer, learning_rate, total_steps):
    warmup_steps = int(0.1 * total_steps)
    decay_steps = total_steps - warmup_steps
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay_scheduler = CosineAnnealingLR(
        optimizer, T_max=decay_steps, eta_min=learning_rate * 0.1
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps],
    )


def select_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def build_dataloader(
    dataset: TextDataset, args: argparse.Namespace, device: torch.device, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    total_val_loss = 0.0

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_val_loss += loss.item()

    model.train()
    return total_val_loss / max(1, len(val_loader))


def main() -> None:
    args = parse_args()
    validate_training_args(args)

    tokenizer = load_tokenizer(args.tokenizer_path)
    model, config = create_model_and_config(args.model_name)
    config.vocab_size = max(tokenizer.unified_vocab.keys()) + 1

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Selected model: {args.model_name}")
    print("Config:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print(f"Model parameter count: {num_params:,}")

    device = select_device()
    print(f"Using device: {device}")

    model.to(device)
    model.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TextDataset(args.dataset_path, tokenizer, config.max_seq_len)
    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset is too short for the configured max_seq_len. "
            "Use a longer dataset or smaller max_seq_len."
        )
    train_loader = build_dataloader(train_dataset, args, device, shuffle=True)

    val_loader: DataLoader | None = None
    if args.val_dataset_path is not None:
        val_dataset = TextDataset(args.val_dataset_path, tokenizer, config.max_seq_len)
        val_loader = build_dataloader(val_dataset, args, device, shuffle=False)

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_scheduler(optimizer, args.learning_rate, total_steps)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            global_step += 1
            # Shape: (batch_size, seq_len)
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)  # Shape: (batch_size, seq_len, vocab_size)
            # Flatten tensors to meet CrossEntropyLoss's expectation
            # logits: (batch_size * seq_len, vocab_size)
            # targets: (batch_size * seq_len,)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.grad_clip_norm
            )
            optimizer.step()
            scheduler.step()

            if global_step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={global_step} batch={batch_idx} "
                    f"loss={loss.item():.4f}"
                )

            if val_loader is not None and global_step % args.val_every == 0:
                # Evaluate the model if the user passes in the validation dataset
                val_loss = validate(model, val_loader, device, criterion)
                print(f"[val] epoch={epoch} step={global_step} loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), args.output_dir / "best_model.pt")
        torch.save(model.state_dict(), args.output_dir / f"model_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
