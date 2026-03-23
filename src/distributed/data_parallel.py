import torch
import torch.distributed as dist
import torch.nn as nn


class DistributedDataParallel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        assert dist.is_initialized(), (
            "Distributed process group must be initialized before wrapping the model."
        )

        self.model = model
        self.num_ranks = dist.get_world_size()

        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param, src=0)
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._average_grad)

        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)

    def _average_grad(self, param: torch.Tensor) -> None:
        assert param.grad is not None
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= self.num_ranks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
