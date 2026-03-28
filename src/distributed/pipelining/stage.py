from collections import deque

import torch
import torch.distributed as dist
import torch.nn as nn


class PipelineStage:
    def __init__(
        self,
        module: nn.Module,
        rank: int,
        world_size: int,
        device: torch.device,
        microbatch_shape: tuple,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            module: The specific neural network chunk assigned to this GPU.
            rank: The global pipeline rank.
            world_size: Total number of pipeline stages.
            device: The physical GPU (e.g., torch.device('cuda:0')).
            microbatch_shape: The physical shape of the tensor crossing the network
                              (e.g., [microbatch_size, seq_len, embed_dim]).
            dtype: The precision format (crucial for network byte matching).
        """
        # --- 1. Core State ---
        self.module = module.to(device)
        self.device = device
        self.dtype = dtype
        self.microbatch_shape = microbatch_shape

        # --- 2. Topology Tracking ---
        self.rank = rank
        self.world_size = world_size

        # Boolean flags make the routing logic much cleaner later.
        self.is_first_stage = rank == 0
        self.is_last_stage = rank == world_size - 1

        # Determine exactly who to talk to.
        self.prev_rank = rank - 1 if not self.is_first_stage else None
        self.next_rank = rank + 1 if not self.is_last_stage else None

        # --- 3. Activation Memory Management ---
        # A simple Python list acts as our FIFO (First-In-First-Out) queue.
        # It will hold tuples of (input_tensor, output_tensor).
        self.activations_queue: deque[tuple[torch.Tensor, torch.Tensor]] = deque()

    def run_forward_step(self, x: torch.Tensor | None = None):
        if not self.is_first_stage:
            x = torch.empty(self.microbatch_shape, dtype=self.dtype, device=self.device)
            dist.recv(x, src=self.prev_rank)
            x.requires_grad_()

        assert x is not None

        out: torch.Tensor = self.module(x)
        self.activations_queue.append((x, out))

        if not self.is_last_stage:
            dist.send(out.detach(), dst=self.next_rank)
        else:
            return out

    def run_backward_step(self, loss: torch.Tensor | None = None) -> None:
        x, out = self.activations_queue.popleft()

        if not self.is_last_stage:
            grad = torch.empty(
                self.microbatch_shape, dtype=self.dtype, device=self.device
            )
            dist.recv(grad, src=self.next_rank)

            out.backward(grad)
        else:
            assert loss is not None
            loss.backward()

        assert x.grad is not None

        if not self.is_first_stage:
            dist.send(x.grad, dst=self.prev_rank)
