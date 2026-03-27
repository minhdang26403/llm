"""Distributed training entrypoint with model/config selection from CLI."""

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import TextDataset
from distributed import DistributedDataParallel, ZeroRedundancyOptimizer
from models.config import GPTConfig, LlamaConfig, ModelConfig
from models.gpt import GPT
from models.llama import Llama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a large language model with custom DDP + ZeRO-1. "
            "Launch with torchrun."
        )
    )
    parser.add_argument("model_name", choices=["gpt", "llama"], help="Model family.")
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Training dataset .bin file (uint32 token ids).",
    )
    parser.add_argument(
        "--val-dataset-path",
        type=Path,
        default=None,
        help="Optional validation dataset .bin file (uint32 token ids).",
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
    parser.add_argument(
        "--backend",
        choices=["gloo", "nccl"],
        default=None,
        help=(
            "Distributed backend. Defaults to nccl when CUDA is available, "
            "otherwise gloo."
        ),
    )
    parser.add_argument(
        "--bucket-cap-mb",
        type=int,
        default=25,
        help="Gradient bucket size in MB for custom DDP reducer.",
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
    if args.bucket_cap_mb <= 0:
        raise ValueError("--bucket-cap-mb must be > 0")
    _validate_bin_file_arg(args.dataset_path, "dataset_path")
    if args.val_dataset_path is not None:
        _validate_bin_file_arg(args.val_dataset_path, "--val-dataset-path")


def _validate_bin_file_arg(path: Path, arg_name: str) -> None:
    if not path.exists() or not path.is_file():
        raise ValueError(f"{arg_name} must be an existing file: {path}")
    if path.suffix != ".bin":
        raise ValueError(f"{arg_name} must be a .bin file")


def create_model_and_config(model_name: str) -> tuple[nn.Module, ModelConfig]:
    if model_name == "gpt":
        config = GPTConfig.default()
        return GPT(config), config
    elif model_name == "llama":
        config = LlamaConfig.default()
        return Llama(config), config
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer, learning_rate: float, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_steps = int(0.1 * total_steps)
    if warmup_steps == 0:
        warmup_steps = 1
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        # Edge case for tiny runs: fallback to linear warmup only.
        return LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
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


def infer_backend(backend_arg: str | None) -> str:
    if backend_arg is not None:
        return backend_arg
    return "nccl" if torch.cuda.is_available() else "gloo"


def init_distributed(backend: str) -> tuple[int, int, torch.device]:
    required_env = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing_env = [name for name in required_env if name not in os.environ]
    if missing_env:
        raise RuntimeError(
            "Missing distributed environment variables. "
            "Launch with torchrun. Missing: " + ", ".join(missing_env)
        )

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA devices.")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, world_size, device


def build_dataloader(
    dataset: TextDataset,
    args: argparse.Namespace,
    device: torch.device,
    sampler: DistributedSampler,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
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
    total_val_loss = torch.tensor(0.0, device=device)
    total_val_batches = torch.tensor(0.0, device=device)

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_val_loss += loss.detach()
        total_val_batches += 1.0

    dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_val_batches, op=dist.ReduceOp.SUM)

    model.train()
    if total_val_batches.item() == 0:
        return float("inf")
    return (total_val_loss / total_val_batches).item()


def main() -> None:  # noqa: C901
    args = parse_args()
    validate_training_args(args)
    backend = infer_backend(args.backend)
    rank, world_size, device = init_distributed(backend)

    try:
        model, config = create_model_and_config(args.model_name)

        num_params = sum(p.numel() for p in model.parameters())
        if rank == 0:
            print(f"Selected model: {args.model_name}")
            print(f"World size: {world_size}")
            print(f"Backend: {backend}")
            print("Config:")
            for key, value in asdict(config).items():
                print(f"  {key}: {value}")
            print(f"Model parameter count: {num_params:,}")
            print(f"Using device: {device}")

        model.to(device)
        model.train()
        ddp_model = DistributedDataParallel(model, bucket_cap_mb=args.bucket_cap_mb)

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        train_dataset = TextDataset(args.dataset_path, config.max_seq_len)
        if len(train_dataset) == 0:
            raise RuntimeError(
                "Training dataset is too short for the configured max_seq_len. "
                "Use a longer dataset or smaller max_seq_len."
            )

        train_sampler: DistributedSampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        train_loader = build_dataloader(train_dataset, args, device, train_sampler)
        if rank == 0:
            print("Number of iterations per epoch per rank:", len(train_loader))

        val_loader = None
        if args.val_dataset_path is not None:
            val_dataset = TextDataset(args.val_dataset_path, config.max_seq_len)
            val_sampler: DistributedSampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
            val_loader = build_dataloader(val_dataset, args, device, val_sampler)

        optimizer = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            AdamW,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        total_steps = len(train_loader) * args.num_epochs
        scheduler = get_scheduler(optimizer.optim, args.learning_rate, total_steps)
        criterion = nn.CrossEntropyLoss()
        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(args.num_epochs):
            train_sampler.set_epoch(epoch)

            for _, (inputs, targets) in enumerate(train_loader):
                # Shape: (batch_size, seq_len)
                inputs, targets = inputs.to(device), targets.to(device)

                logits = ddp_model(inputs)  # Shape: (batch_size, seq_len, vocab_size)
                # Flatten tensors to meet CrossEntropyLoss's expectation
                # logits: (batch_size * seq_len, vocab_size)
                # targets: (batch_size * seq_len,)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    ddp_model.parameters(), max_norm=args.grad_clip_norm
                )
                optimizer.step()
                scheduler.step()

                global_step += 1
                if global_step % args.log_every == 0:
                    reduced_loss = loss.detach().clone()
                    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                    reduced_loss /= world_size
                    if rank == 0:
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"loss={reduced_loss.item():.4f}"
                        )

                if val_loader is not None and global_step % args.val_every == 0:
                    val_loss = validate(ddp_model, val_loader, device, criterion)
                    if rank == 0:
                        print(
                            f"[val] epoch={epoch} step={global_step} "
                            f"loss={val_loss:.4f}"
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(
                                ddp_model.model.state_dict(),
                                args.output_dir / "best_model.pt",
                            )

            if rank == 0:
                torch.save(
                    ddp_model.model.state_dict(),
                    args.output_dir / f"model_epoch_{epoch}.pt",
                )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
