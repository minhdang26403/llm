import torch
import torch.distributed as dist


class _CopyToModelParallelRegion(torch.autograd.Function):
    """
    Passes input through in the forward pass, but all-reduces gradients in the
    backward pass.
    """

    @staticmethod
    def forward(ctx, input_tensor, tp_group):
        # Stash the process group in the context so the backward pass can use it
        ctx.tp_group = tp_group

        # Forward pass is an Identity operation
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # If no process group, we are running on a single GPU (fallback)
        if ctx.tp_group is None:
            return grad_output, None

        # Backward pass requires summing the gradients across the TP group
        # We clone the gradient to avoid mutating the autograd graph's tensors in-place
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.tp_group)

        # Return gradients for the inputs (input_tensor, tp_group).
        # tp_group is not a tensor, so its gradient is None.
        return grad_input, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    All-reduces inputs in the forward pass, but passes gradients through in the
    backward pass.
    """

    @staticmethod
    def forward(ctx, input_tensor, tp_group):
        # Forward pass requires summing the partial activations
        if tp_group is None:
            return input_tensor

        # We clone to ensure we don't accidentally overwrite the input tensor in memory
        output = input_tensor.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is an Identity operation
        return grad_output, None


def copy_to_tensor_model_parallel_region(input_tensor, tp_group):
    """
    Use this right before a ColumnParallelLinear layer.
    """
    return _CopyToModelParallelRegion.apply(input_tensor, tp_group)


def reduce_from_tensor_model_parallel_region(input_tensor, tp_group):
    """
    Use this right after a RowParallelLinear layer's matrix multiplication.
    """
    return _ReduceFromModelParallelRegion.apply(input_tensor, tp_group)
