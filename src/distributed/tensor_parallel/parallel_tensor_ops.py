from typing import Any

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


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input_tensor: torch.Tensor, tp_group: dist.ProcessGroup
    ) -> torch.Tensor:
        if tp_group is None:
            return input_tensor

        ctx.tp_group = tp_group

        transposed_input = input_tensor.transpose(0, 1).contiguous()
        partition_seq_len, batch_size, embed_dim = transposed_input.shape

        world_size = dist.get_world_size(group=tp_group)
        full_seq_len = partition_seq_len * world_size

        output = torch.empty(
            (full_seq_len, batch_size, embed_dim),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

        dist.all_gather_into_tensor(output, transposed_input, group=tp_group)

        return output.transpose(0, 1).contiguous()

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, Any]:
        grad_output = grad_outputs[0]
        tp_group = ctx.tp_group
        if tp_group is None:
            return grad_output, None

        transposed_grad = grad_output.transpose(0, 1).contiguous()
        full_seq_len, batch_size, embed_dim = transposed_grad.shape

        world_size = dist.get_world_size(group=tp_group)
        partition_seq_len = full_seq_len // world_size

        partitioned_grad = torch.empty(
            (partition_seq_len, batch_size, embed_dim),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        dist.reduce_scatter_tensor(
            partitioned_grad,
            transposed_grad,
            op=dist.ReduceOp.SUM,
            group=tp_group,
        )

        return partitioned_grad.transpose(0, 1).contiguous(), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input_tensor: torch.Tensor, tp_group: dist.ProcessGroup
    ) -> torch.Tensor:
        if tp_group is None:
            return input_tensor

        ctx.tp_group = tp_group

        # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        transposed_input = input_tensor.transpose(0, 1).contiguous()

        seq_len, batch_size, embed_dim = transposed_input.shape
        partition_seq_len = seq_len // dist.get_world_size(group=tp_group)

        output = torch.empty(
            (partition_seq_len, batch_size, embed_dim),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        dist.reduce_scatter_tensor(
            output,
            transposed_input,
            op=dist.ReduceOp.SUM,
            group=tp_group,
        )

        # Transpose back to standard format: (batch, partition_seq_len, hidden_dim)
        return output.transpose(0, 1).contiguous()

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, Any]:
        grad_output = grad_outputs[0]
        tp_group = ctx.tp_group
        if tp_group is None:
            return grad_output, None

        transposed_grad = grad_output.transpose(0, 1).contiguous()
        partition_seq_len, batch_size, embed_dim = transposed_grad.shape

        world_size = dist.get_world_size(group=tp_group)
        full_seq_len = partition_seq_len * world_size

        gathered_grad = torch.empty(
            (full_seq_len, batch_size, embed_dim),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        dist.all_gather_into_tensor(gathered_grad, transposed_grad, group=tp_group)

        return gathered_grad.transpose(0, 1).contiguous(), None


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


def gather_from_sequence_parallel_region(input_tensor, tp_group):
    """
    Use this right before a ParallelAttention layer.
    """
    return _GatherFromSequenceParallelRegion.apply(input_tensor, tp_group)


def reduce_scatter_to_sequence_parallel_region(input_tensor, tp_group):
    """
    Use this right after a ParallelAttention layer.
    """
    return _ReduceScatterToSequenceParallelRegion.apply(input_tensor, tp_group)
