from enum import Enum, auto

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

    def unshard(self, compute_dtype=torch.float16, post_backward_hook_fn=None):
        if self.local_shard is None:
            return

        local_shard_compute = self.local_shard.to(compute_dtype)
        gathered_params = torch.zeros(
            self.local_shard.numel() * self.world_size,
            dtype=compute_dtype,
            device=self.local_shard.device,
        )
        dist.all_gather_into_tensor(gathered_params, local_shard_compute)

        # We tell Autograd to explicitly track this 1D buffer. The gradient for this
        # buffer is computed after all the gradients of local shards are computed.
        gathered_params.requires_grad_()
        if post_backward_hook_fn:
            # The hook will trigger to reduce gradients across GPUs.
            gathered_params.register_hook(post_backward_hook_fn)

        if self.pad_len > 0:
            gathered_params = gathered_params[: -self.pad_len]

        start_idx = 0
        for name, shape, dtype, numel in self.param_metadata:
            end_idx = start_idx + numel
            param = gathered_params[start_idx:end_idx].view(shape).to(dtype)
            setattr(self.module, name, param)
            start_idx += numel

    def reshard(self):
        if self.local_shard is None:
            return

        for name, _, _, _ in self.param_metadata:
            if hasattr(self.module, name):
                delattr(self.module, name)
                setattr(self.module, name, None)

    def reduce_scatter_gradients(self, full_grad):
        """
        Receives the full gradients (for this data partition), reduces them across GPUs,
        and scatters the 1/N chunk (partitioned by parameters) to our local master
        shard.
        """

        # 1. We need an empty buffer to hold our specific 1/N gradient chunk.
        # It must match the shape and dtype of self.local_shard.
        local_grad_chunk = torch.zeros_like(self.local_shard)

        # 2. dist.reduce_scatter_tensor requires both the input (full_grad)
        # and output (local_grad_chunk) to be the exact same dtype.
        full_grad = full_grad.to(self.local_shard.dtype)

        # 3. Reduce the gradients across data partitions and scatter the correct chunk
        # of gradients for this local shard.
        chunks = list(torch.chunk(full_grad, self.world_size))
        dist.reduce_scatter_tensor(local_grad_chunk, chunks)

        # 4. Assign the resulting chunk directly to our master weight's .grad attribute
        self.local_shard.grad = local_grad_chunk


class ShardingStrategy(Enum):
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()


class FullyShardedDataParallel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    ):
        super().__init__()

        self.module = module
        self.sharding_strategy = sharding_strategy

        self.flat_param = FlatParameter(module)

        self._register_forward_hooks()

    def _register_forward_hooks(self):
        """
        Registers the pre-forward and post-forward hooks on self.module
        to trigger memory unsharding and resharding.
        """

        def forward_pre_hook(module, input):
            # 1. Define the tensor hook that will run during loss.backward()
            def post_backward_tensor_hook(full_grad):
                self.flat_param.reduce_scatter_gradients(full_grad)
                self.flat_param.reshard()

            # 2. Pass this hook down into unshard so it gets attached to the buffer
            self.flat_param.unshard(post_backward_hook_fn=post_backward_tensor_hook)

        self.module.register_forward_pre_hook(forward_pre_hook)

        def forward_hook(module, input, output):
            if self.sharding_strategy == ShardingStrategy.FULL_SHARD:
                self.flat_param.reshard()

        self.module.register_forward_hook(forward_hook)

    def _register_backward_hooks(self):
        def backward_pre_hook(module, grad_output):
            if self.sharding_strategy == ShardingStrategy.FULL_SHARD:
                # Only in case of FULL_SHARD that we reshard the parameters after the
                # forward pass. Hence, we need to unshard before the backward pass.
                self.flat_param.unshard()

        self.module.register_full_backward_pre_hook(backward_pre_hook)

    def forward(self, *args, **kwargs):
        # We simply pass the inputs down to the inner module.
        # The hooks we registered will automatically intercept the math.
        return self.module(*args, **kwargs)
