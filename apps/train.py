import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from distributed import DistributedDataParallel


# 1. A simple dummy model to test distributed training
class DummyLLM(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=6):
        super().__init__()
        # Create multiple distinct layers so we have lots of separate parameter tensors
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return self.fc_out(x)


def main():
    # 2. Initialize the distributed environment
    # torchrun automatically sets these environment variables
    dist.init_process_group(backend="gloo")  # Use 'gloo' for CPU-to-CPU communication

    global_rank = int(os.environ["RANK"])

    # 3. CRITICAL: Bind this specific process to a specific GPU
    # torch.cuda.set_device(local_rank)
    # device = torch.device(f"cuda:{local_rank}")

    # No GPU, so use CPU instead
    device = torch.device("cpu")

    # 4. Instantiate model
    hidden_dim = 512
    print(f"[Rank {global_rank}] Initializing model...")
    model = DummyLLM(hidden_dim=hidden_dim).to(device)

    # Force a tiny bucket size to trigger the multi-bucket codepath
    ddp_model = DistributedDataParallel(model, bucket_cap_mb=2)

    # 5. Setup Optimizer and Dummy Data
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)

    # Ensure each rank gets DIFFERENT data (micro-batches)
    torch.manual_seed(global_rank)
    batch_size = 32
    hidden_dim = 512
    dummy_inputs = torch.randn(batch_size, hidden_dim).to(device)
    dummy_targets = torch.randint(0, 10, (batch_size,)).to(device)
    criterion = nn.CrossEntropyLoss()

    # 6. The Training Loop
    print(f"[Rank {global_rank}] Starting training loop...")
    for step in range(3):
        optimizer.zero_grad()

        outputs = ddp_model(dummy_inputs)
        loss = criterion(outputs, dummy_targets)

        loss.backward()  # Our hooks fire here!

        optimizer.step()

        print(f"[Rank {global_rank}] Step {step} | Loss: {loss.item():.4f}")

    # Clean up the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
