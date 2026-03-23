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
        self.param_map: dict[int, tuple[torch.Tensor, tuple, int, int]] = {}

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

        for _, param_info in self.param_map.items():
            grad, shape, start, end = param_info
            grad.copy_(active_bucket[start:end].view(shape))

        # Reset bucket state
        self.bucket_len = 0
        self.param_map = {}

    def _average_grad(self, param: torch.Tensor) -> None:
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
            self.bucket[start:end].copy_(param.grad.view(-1))

            self.param_map[id(param)] = (
                param.grad,
                param.shape,
                start,
                end,
            )
            self.bucket_len = end

        # If this is the absolute last gradient of the backward pass, force a flush
        if self.ready_grads == self.expected_grads:
            self._flush_bucket()
            self.ready_grads = 0  # Reset the counter for the next training step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
