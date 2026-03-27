from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


class ZeroRedundancyOptimizer:
    """
    A PyTorch Optimizer wrapper that implements ZeRO-1 (Optimizer State Partitioning).

    This wrapper dramatically reduces the memory footprint of training large models
    by partitioning the optimizer states (e.g., Adam's momentum and variance) across
    all available distributed ranks. Instead of every GPU holding 100% of the FP32
    master weights and optimizer states, each GPU only holds a `1/world_size` fraction.

    Note:
        This wrapper requires the model to be wrapped in a DistributedDataParallel (DDP)
        module that performs an `AllReduce` on the gradients during the backward pass.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        optimizer_class: type[optim.Optimizer],
        **defaults,
    ):
        """
        Initializes the ZeRO-1 optimizer wrapper.

        This method flattens the provided parameters into a single contiguous 1D tensor,
        casts it to FP32 to serve as the master weights, pads it for even network
        distribution, and assigns a specific shard to the current distributed rank.

        Args:
            params (iterable): An iterable of `torch.Tensor` objects
                (usually `model.parameters()`) that require optimization.
            optimizer_class (type[torch.optim.Optimizer]): The class of the inner
                optimizer to instantiate (e.g., `torch.optim.Adam`, `torch.optim.SGD`).
            **defaults: Keyword arguments containing the hyperparameters to pass
                to the inner `optimizer_class` (e.g., `lr=1e-3`, `weight_decay=0.01`).

        Raises:
            AssertionError: If the distributed process group has not been initialized.
        """

        # 1. Setup Distributed Context
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 2. Extract and store references to the original model parameters
        # We need these later to push the updated weights back into the model
        self.model_params = list(params)

        # 3. Flatten the model parameters into a single 1D tensor.
        flat_params = nn.utils.parameters_to_vector(self.model_params)

        # 4. Calculate Padding
        # The network requires equal sized chunks across all GPUs.
        remainder = flat_params.numel() % self.world_size
        self.pad_len = self.world_size - remainder if remainder > 0 else 0

        if self.pad_len > 0:
            pad_tensor = torch.zeros(
                self.pad_len, dtype=flat_params.dtype, device=flat_params.device
            )
            flat_params = torch.cat([flat_params, pad_tensor])

        # 5. Calculate shard indices for this specific rank
        shard_size = flat_params.numel() // self.world_size
        self.start_idx = self.rank * shard_size
        self.end_idx = self.start_idx + shard_size

        # 6. Extract the local shard
        # This creates a view to the original tensor, not a copy.
        local_slice = flat_params[self.start_idx : self.end_idx]

        # We detach the tensor to break it off from the model's autograd graph.
        # Still shares memory with flat_params.
        detached_slice = local_slice.detach()

        # Cast to FP32 because ZeRO-1 requires high-precision "master weights" for the
        # optimizer to update safely.
        self.local_param = detached_slice.float()
        self.local_param.requires_grad = True

        # 7. Initialize the underlying optimizer (e.g., optim.Adam)
        # Notice we ONLY pass it our specific 1D slice, drastically reducing memory.
        self.optim = optimizer_class([self.local_param], **defaults)

        # We expose param_groups to comply with PyTorch's standard Optimizer API
        self.param_groups = self.optim.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step and synchronizes the updated parameters.

        This method executes the following data pipeline:
        1. Gathers and flattens the native gradients from the model parameters.
        2. Slices out the specific chunk of gradients assigned to this rank.
        3. Casts the local gradient slice to FP32 and feeds it to the inner optimizer.
        4. Calls `step()` on the inner optimizer to update the local master weights.
        5. Casts the updated shard back to the native model precision.
        6. Performs an `AllGather` to broadcast the updated shards across all ranks.
        7. Un-flattens the gathered 1D tensor back into the original model parameters.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Optional for most optimizers.

        Returns:
            Optional[float]: The loss returned by the closure, if provided;
                otherwise, None.


        Optimization:
        Note that in this implementation, we only update the model's parameters after
        all of their gradients are computed. However, we can overlap the gradient
        computation with the weight update.

        Specifically, we can use Bucket-Level Stepping:
        1. Bucketing: Instead of flattening the whole model, we flatten parameters into
            smaller chunks (e.g., 25MB buckets)
        2. Autograd Hooks: Attach a post-backward Autograd hook to each parameter.
        3. Accumulate tensors into a bucket: Once a parameter's gradient is computed,
            the hook is called, and we add the gradient into the bucket.
        4. Gradient Sync: When the bucket reaches its capacity, we use reduction
            operation to compute the averaged gradient across all GPUs.
        5.  Micro-Step: Call a micro-optimizer step just for that specific 25MB bucket.
        """

        # 1. Flatten all individual gradients into a single list
        grad_list = []
        for p in self.model_params:
            if p.grad is None:
                # If a parameter wasn't used in the forward pass, pad it with zeros
                grad_list.append(torch.zeros_like(p.view(-1)))
            else:
                grad_list.append(p.grad.view(-1))

        # 2. Concatenate into a single massive 1D gradient tensor
        flat_grads = torch.cat(grad_list)

        # 3. Apply the exact same padding we used for the weights in __init__
        if self.pad_len > 0:
            pad_tensor = torch.zeros(
                self.pad_len, dtype=flat_grads.dtype, device=flat_grads.device
            )
            flat_grads = torch.cat([flat_grads, pad_tensor])

        # 4. Extract this rank's specific gradient slice and hand the gradient slice to
        # our FP32 master weight
        self.local_param.grad = flat_grads[self.start_idx : self.end_idx].float()

        # 5. Step the inner optimizer, which updates `self.local_param` in-place.
        loss = self.optim.step(closure)

        # 6. Cast our freshly updated FP32 shard back down to the model's native dtype
        updated_local_param_native = self.local_param.to(flat_grads.dtype)

        # 7. Allocate an empty buffer large enough to hold the full 1D model
        gathered_params = torch.zeros_like(flat_grads)

        # 8. AllGather: Every GPU broadcasts its updated shard to everyone else.
        # This reconstructs the complete, updated 1D model on every rank.
        dist.all_gather_into_tensor(gathered_params, updated_local_param_native)

        # 9. Strip off the padding at the end
        if self.pad_len > 0:
            gathered_params = gathered_params[: -self.pad_len]

        # 10. Unpack the full 1D tensor back into the model's actual 2D/1D parameters
        nn.utils.vector_to_parameters(gathered_params, self.model_params)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Clears the gradients of all optimized parameters.

        This method ensures that gradients are cleared in both the original model's
        leaf tensors and the inner optimizer's 1D shard to prevent VRAM leaks during
        the subsequent forward pass.

        Args:
            set_to_none (bool, optional): If True, sets the `.grad` attribute to None
                instead of filling the existing memory buffer with zeros.
                This significantly lowers peak VRAM usage. Defaults to True.
        """
        for p in self.model_params:
            if p.grad is not None:
                if set_to_none:
                    # This is the modern PyTorch default. It saves memory by completely
                    # deleting the gradient tensor rather than filling it with zeros.
                    p.grad = None
                else:
                    # Legacy behavior: keeps the tensor allocated but zeros the memory
                    p.grad.zero_()

        # Delegate this to the inner optimizer, which will look
        # at self.local_param and clear its .grad attribute.
        self.optim.zero_grad(set_to_none)
