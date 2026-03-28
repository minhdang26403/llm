import torch
import torch.distributed as dist
import torch.nn as nn

from .parallel_tensor_ops import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)


class ColumnParallelLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, tp_group: dist.ProcessGroup
    ):
        super().__init__()

        self.tp_group = tp_group
        self.tp_world_size = dist.get_world_size(group=tp_group)

        assert out_features % self.tp_world_size == 0, (
            f"out_features ({out_features}) must be divisible by TP world size "
            f"({self.tp_world_size})"
        )

        self.out_features_per_partition = out_features // self.tp_world_size
        self.linear = nn.Linear(
            in_features, self.out_features_per_partition, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel = copy_to_tensor_model_parallel_region(x, self.tp_group)
        return self.linear(x_parallel)


class RowParallelLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, tp_group: dist.ProcessGroup
    ):
        super().__init__()

        self.tp_group = tp_group
        self.tp_world_size = dist.get_world_size(group=tp_group)

        assert in_features % self.tp_world_size == 0, (
            f"in_features ({in_features}) must be divisible by TP world size "
            f"({self.tp_world_size})"
        )

        self.in_features_per_partition = in_features // self.tp_world_size
        self.linear = nn.Linear(
            self.in_features_per_partition, out_features, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_parallel = self.linear(x)
        return reduce_from_tensor_model_parallel_region(out_parallel, self.tp_group)
