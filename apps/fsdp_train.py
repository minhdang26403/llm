import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from distributed import FullyShardedDataParallel, ShardingStrategy


class SimpleTransformerBlock(nn.Module):
    """A dummy module representing a layer we want to shard."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x):
        return self.ffn(x)


def main():
    # 1. Initialize the distributed environment (torchrun handles the env vars)
    dist.init_process_group(backend="gloo")
    rank = int(os.environ["RANK"])

    # Set manual seed so all ranks initialize with the exact same weights
    torch.manual_seed(42)

    # 2. Build the model
    hidden_dim = 16

    # We wrap individual blocks in FSDP to achieve compute/communication overlap!
    # Let's test ZeRO-3 (FULL_SHARD) first.
    strategy = ShardingStrategy.FULL_SHARD

    model = nn.Sequential(
        FullyShardedDataParallel(
            SimpleTransformerBlock(hidden_dim), sharding_strategy=strategy
        ),
        FullyShardedDataParallel(
            SimpleTransformerBlock(hidden_dim), sharding_strategy=strategy
        ),
    )

    # 3. Setup Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. Dummy Data
    batch_size = 4
    x = torch.randn(batch_size, hidden_dim)
    x.requires_grad_()
    target = torch.randn(batch_size, hidden_dim)
    criterion = nn.MSELoss()

    if rank == 0:
        print(f"Setup complete. Strategy: {strategy.name}. Starting training loop...")

    # 5. Training Loop
    for step in range(3):
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        if rank == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

    # Inspect the local shard on Rank 0 to prove it updated
    if rank == 0:
        shard = list(model.parameters())[0]
        print(f"\n[Rank 0] Success! Final local shard shape: {shard.shape}")
        print(f"[Rank 0] Final local shard requires_grad: {shard.requires_grad}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
