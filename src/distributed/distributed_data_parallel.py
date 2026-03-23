import contextlib
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn


class DistributedDataParallel(nn.Module):
    def __init__(self, model: nn.Module, bucket_cap_mb: int = 25):
        super().__init__()

        assert dist.is_initialized(), (
            "Distributed process group must be initialized before wrapping the model."
        )

        self.model = model
        self.num_ranks = dist.get_world_size()

        # Dynamically grab the device from the first parameter
        first_param = next(model.parameters())
        self.device = first_param.device

        self.bucket_max_elements = bucket_cap_mb * 1024 * 1024 // 4
        self.bucket = torch.zeros(
            self.bucket_max_elements, device=self.device, dtype=torch.float
        )
        self.bucket_len = 0
        self.bucket_mapping: list[tuple[torch.Tensor, torch.Tensor]] = []

        self.require_backward_grad_sync = True

        self.expected_grads = 0
        self.ready_grads = 0

        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param, src=0)
                if param.requires_grad:
                    self.expected_grads += 1
                    param.register_post_accumulate_grad_hook(self._average_grad)

        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)

    def _flush_bucket(self) -> None:
        """Helper method to sync and unpack the current bucket."""
        if self.bucket_len == 0:
            return

        # Only sync the active portion of the bucket, not the trailing zeros
        active_bucket = self.bucket[: self.bucket_len]
        dist.all_reduce(active_bucket, op=dist.ReduceOp.SUM)
        active_bucket /= self.num_ranks

        for grad, bucket_view in self.bucket_mapping:
            grad.copy_(bucket_view)

        # Reset bucket state
        self.bucket_len = 0
        self.bucket_mapping.clear()

    def _average_grad(self, param: torch.Tensor) -> None:
        # If no_sync is active, just return. PyTorch will naturally accumulate the
        # gradient in param.grad locally.
        if not self.require_backward_grad_sync:
            return

        assert param.grad is not None

        self.ready_grads += 1
        total_size = param.grad.numel()

        if total_size > self.bucket_max_elements:
            # If the tensor is massive, bypass the bucket entirely and sync it immediately
            fp32_grad = param.grad.float()
            dist.all_reduce(fp32_grad, op=dist.ReduceOp.SUM)
            fp32_grad /= self.num_ranks
            param.grad.copy_(fp32_grad)
        else:
            # If adding this tensor overflows the CURRENT bucket, flush it first
            if self.bucket_len + total_size > self.bucket_max_elements:
                self._flush_bucket()

            # Now, safely add the gradient to the bucket
            start = self.bucket_len
            end = self.bucket_len + total_size

            # Create a view of the bucket that shares the same shape as the gradient
            bucket_view = self.bucket[start:end].view_as(param.grad)

            # Copy the local gradient INTO the bucket
            bucket_view.copy_(param.grad)

            # Store the tuple (destination_grad, source_view) in our list
            self.bucket_mapping.append((param.grad, bucket_view))
            self.bucket_len = end

        # If this is the absolute last gradient of the backward pass, force a flush
        if self.ready_grads == self.expected_grads:
            self._flush_bucket()
            self.ready_grads = 0  # Reset the counter for the next training step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @contextlib.contextmanager
    def no_sync(self) -> Iterator[None]:
        """
        Context manager to disable gradient synchronization across ranks.
        Useful for gradient accumulation to prevent network overhead.
        """
        old_sync_status = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_sync_status
