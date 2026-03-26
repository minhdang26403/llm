import contextlib
from collections import defaultdict
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
        self.world_size = dist.get_world_size()

        # Dynamically grab the device from the first parameter
        first_param = next(model.parameters())
        self.device = first_param.device

        self.bucket_max_elements = bucket_cap_mb * 1024 * 1024 // 4
        self.bucket = torch.zeros(
            self.bucket_max_elements, device=self.device, dtype=torch.float
        )
        self.bucket_len = 0
        self.bucket_index: list[tuple[torch.Tensor, torch.Tensor]] = []

        self.require_backward_grad_sync = True

        self.expected_grads = 0
        self.ready_grads = 0

        # Can't modify a leaf tensor in place, so we need to use no_grad.
        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param, src=0)  # broadcast is an in-place operation
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
        # Use SUM instead of AVG since ReduceOp.AVG is not supported in Gloo backend.
        dist.all_reduce(active_bucket, op=dist.ReduceOp.SUM)
        active_bucket /= self.world_size

        for grad, bucket_view in self.bucket_index:
            grad.copy_(bucket_view)

        # Reset bucket state
        self.bucket_len = 0
        self.bucket_index.clear()

    def _average_grad(self, param: torch.Tensor) -> None:
        # If no_sync is active, just return. PyTorch will naturally accumulate the
        # gradient in param.grad locally.
        if not self.require_backward_grad_sync:
            return

        assert param.grad is not None

        self.ready_grads += 1
        total_size = param.grad.numel()

        if total_size > self.bucket_max_elements:
            # If the tensor is massive, bypass the bucket entirely and sync it
            # immediately.
            fp32_grad = param.grad.float()
            dist.all_reduce(fp32_grad, op=dist.ReduceOp.SUM)
            fp32_grad /= self.world_size
            param.grad.copy_(fp32_grad)
        else:
            # If adding this tensor overflows the CURRENT bucket, flush it first.
            if self.bucket_len + total_size > self.bucket_max_elements:
                self._flush_bucket()

            # Now, safely add the gradient to the bucket.
            start = self.bucket_len
            end = self.bucket_len + total_size

            # Create a view of the bucket that shares the same shape as the gradient.
            bucket_view = self.bucket[start:end].view_as(param.grad)

            # Copy the local gradient INTO the bucket.
            bucket_view.copy_(param.grad)

            # Store the tuple (destination_grad, source_view) in our bucket index.
            self.bucket_index.append((param.grad, bucket_view))
            self.bucket_len = end

        # If this is the absolute last gradient of the backward pass, force a flush.
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


class DistributedDataParallelStaticBucket(nn.Module):
    def __init__(self, model: nn.Module, bucket_cap_mb: int = 25):
        super().__init__()

        assert dist.is_initialized(), (
            "Distributed process group must be initialized before wrapping the model."
        )

        self.model = model
        self.world_size = dist.get_world_size()
        self.device = next(model.parameters()).device
        self.require_backward_grad_sync = True

        self.bucket_max_elements = bucket_cap_mb * 1024 * 1024 // 4

        # --- STATIC BUCKETING DATA STRUCTURES ---
        self.buckets: list[torch.Tensor] = []  # List of FP32 tensor buffers
        self.bucket_expected_grads: list[int] = []  # How many params in each bucket
        self.bucket_ready_grads: list[int] = []  # Running tally during backward pass

        self.param_to_bucket_info: dict[
            map, tuple[int, int, int]
        ] = {}  # id(param) -> (bucket_idx, start, end)
        self.bucket_idx_to_params: defaultdict[int, list[torch.Tensor]] = defaultdict(
            list
        )  # bucket_idx -> [params]

        self._build_static_buckets()

        # Broadcast initial weights so all GPUs start identically
        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param, src=0)

            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)

    def _build_static_buckets(self) -> None:
        """Pre-calculates memory layouts based on reverse execution order."""
        # 1. Grab params and reverse them to match backward pass order
        params_with_grad = [p for p in self.model.parameters() if p.requires_grad]
        reversed_params = reversed(params_with_grad)

        current_bucket_idx = 0
        current_bucket_size = 0
        current_bucket_expected = 0

        for param in reversed_params:
            numel = param.numel()

            # If adding this param exceeds the cap (and the bucket isn't empty)
            # we "close" the current bucket and move to the next.
            if (
                current_bucket_size + numel > self.bucket_max_elements
                and current_bucket_size > 0
            ):
                self._finalize_bucket(current_bucket_size, current_bucket_expected)
                current_bucket_idx += 1
                current_bucket_size = 0
                current_bucket_expected = 0

            # Map the parameter to its static offset
            start = current_bucket_size
            end = start + numel
            self.param_to_bucket_info[id(param)] = (current_bucket_idx, start, end)
            self.bucket_idx_to_params[current_bucket_idx].append(param)

            current_bucket_size += numel
            current_bucket_expected += 1

            param.register_post_accumulate_grad_hook(self._average_grad)

    def _finalize_bucket(self, size: int, expected) -> None:
        """Allocates the physical memory for a planned bucket."""
        new_bucket = torch.tensor(size, dtype=torch.float, device=self.device)
        self.buckets.append(new_bucket)
        self.bucket_expected_grads.append(expected)
        self.bucket_ready_grads.append(0)

    def _sync_and_unpack_bucket(self, bucket_idx: int) -> None:
        """Fires AllReduce for a specific bucket and writes data back to params."""
        bucket = self.buckets[bucket_idx]

        # 1. Network Sync
        dist.all_reduce(bucket, op=dist.ReduceOp.SUM)
        bucket /= self.world_size

        # 2. Unpack back into the original gradients
        for param in self.bucket_idx_to_params[bucket_idx]:
            _, start, end = self.param_to_bucket_info[id(param)]
            assert param.grad is not None
            param.grad.copy_(bucket[start:end].view(param.shape))

        # 3. Reset the ready counter for the next training step
        self.bucket_ready_grads[bucket_idx] = 0

    def _average_grad(self, param: torch.Tensor) -> None:
        if not self.require_backward_grad_sync:
            return

        assert param.grad is not None

        bucket_idx, start, end = self.param_to_bucket_info[id(param)]

        # Blindly copy the gradient into its pre-assigned slot
        self.buckets[bucket_idx][start:end].copy_(param.grad.view(-1))

        # Increment the counter for this specific bucket
        self.bucket_ready_grads[bucket_idx] += 1

        # If this bucket has received all its expected gradients, fire the sync.
        if (
            self.bucket_ready_grads[bucket_idx]
            == self.bucket_expected_grads[bucket_idx]
        ):
            self._sync_and_unpack_bucket(bucket_idx)

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
