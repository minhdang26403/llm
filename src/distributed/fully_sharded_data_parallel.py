import torch
import torch.distributed as dist
import torch.nn as nn


class FlatParameter(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()

        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 1. Metadata tracking
        self.param_metadata = []
        params_to_flatten = []

        # Extract parameter info before we destroy the module's attributes
        for name, param in module.named_parameters(recurse=False):
            if param is not None:
                self.param_metadata.append(
                    (name, param.shape, param.dtype, param.numel())
                )
                params_to_flatten.append(param)

        if not params_to_flatten:
            self.local_shard = None
            return

        # 2. Flatten, Pad, and Shard
        flat_params = nn.utils.parameters_to_vector(params_to_flatten)
        remainder = flat_params.numel() % self.world_size
        self.pad_len = self.world_size - remainder if remainder > 0 else 0

        if self.pad_len > 0:
            pad_tensor = torch.zeros(
                self.pad_len, dtype=flat_params.dtype, device=flat_params.device
            )
            flat_params = torch.cat([flat_params, pad_tensor])

        shard_size = flat_params.numel() // self.world_size
        start_idx = self.rank * shard_size
        end_idx = start_idx + shard_size

        # Create the local weight shard and register it as a parameter of the
        # FlatParameter class so the optimizer can find it later
        self.local_shard = nn.Parameter(flat_params[start_idx:end_idx].clone().detach())

        for name, _, _, _ in self.param_metadata:
            # Delete the original parameter to free the VRAM
            delattr(self.module, name)
            # Leave a breadcrumb so the forward pass doesn't instantly crash before our
            # hooks can rebuild it
            setattr(self.module, name, None)

    def unshard(self, compute_dtype=torch.float16):
        if not self.local_shard:
            return

        local_shard_compute = self.local_shard.to(compute_dtype)
        gathered_params = torch.zeros(
            self.local_shard.numel() * self.world_size,
            dtype=compute_dtype,
            device=self.local_shard.device,
        )
        dist.all_gather_into_tensor(gathered_params, local_shard_compute)

        if self.pad_len > 0:
            gathered_params = gathered_params[: -self.pad_len]

        start_idx = 0
        for name, shape, dtype, numel in self.param_metadata:
            end_idx = start_idx + numel
            param = gathered_params[start_idx:end_idx].view(shape).to(dtype)
            setattr(self.module, name, param)
            start_idx += numel

    def reshard(self):
        if not self.local_shard:
            return

        for name, _, _, _ in self.param_metadata:
            if hasattr(self.module, name):
                delattr(self.module, name)
                setattr(self.module, name, None)
