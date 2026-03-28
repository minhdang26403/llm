import torch
import torch.nn.functional as F

from .stage import PipelineStage


class ScheduleGPipe:
    def __init__(self, stage: PipelineStage, num_microbatches: int):
        self.stage = stage
        self.num_microbatches = num_microbatches

    def step(
        self,
        x_global: torch.Tensor | None = None,
        target_global: torch.Tensor | None = None,
    ):

        x_microbatches: tuple[torch.Tensor, ...] = ()
        target_microbatches: tuple[torch.Tensor, ...] = ()

        if self.stage.is_first_stage:
            assert x_global is not None
            x_microbatches = x_global.chunk(self.num_microbatches)

        if self.stage.is_last_stage:
            assert target_global is not None
            target_microbatches = target_global.chunk(self.num_microbatches)

        forward_logits = []
        for i in range(self.num_microbatches):
            x = x_microbatches[i] if self.stage.is_first_stage else None
            logits = self.stage.run_forward_step(x)
            if self.stage.is_last_stage:
                forward_logits.append(logits)

        total_loss = 0.0
        for i in reversed(range(self.num_microbatches)):
            loss = None
            if self.stage.is_last_stage:
                logits = forward_logits[i]
                target = target_microbatches[i]

                loss = F.cross_entropy(logits, target)
                total_loss += loss.item()

            self.stage.run_backward_step(loss)

        if self.stage.is_last_stage:
            return total_loss / self.num_microbatches

        return None


class Schedule1F1B:
    def __init__(self, stage: PipelineStage, num_microbatches: int):
        self.stage = stage
        self.num_microbatches = num_microbatches
        self.forward_logits: list[torch.Tensor] = []
        self.total_loss = 0.0

    def _run_forward_step(self, x_microbatches: tuple[torch.Tensor, ...], i: int):
        x = x_microbatches[i] if self.stage.is_first_stage else None
        logits = self.stage.run_forward_step(x)

        if self.stage.is_last_stage:
            self.forward_logits.append(logits)

    def _run_backward_step(self, target_microbatches: tuple[torch.Tensor, ...], i: int):
        loss = None

        if self.stage.is_last_stage:
            logits = self.forward_logits[i]
            target = target_microbatches[i]
            loss = F.cross_entropy(logits, target)
            self.total_loss += loss.item()

        self.stage.run_backward_step(loss)

    def step(
        self,
        x_global: torch.Tensor | None = None,
        target_global: torch.Tensor | None = None,
    ):
        x_microbatches: tuple[torch.Tensor, ...] = ()
        target_microbatches: tuple[torch.Tensor, ...] = ()

        if self.stage.is_first_stage:
            assert x_global is not None
            x_microbatches = x_global.chunk(self.num_microbatches)

        if self.stage.is_last_stage:
            assert target_global is not None
            target_microbatches = target_global.chunk(self.num_microbatches)

        # The number of pipeline stages between this GPU and the end of the pipeline.
        # Stage 0 has the most warmups. The final stage has 0 warmups.
        distance_to_end = self.stage.world_size - 1 - self.stage.rank
        # We cap warmups at num_microbatches in case the batch is unusually small
        num_warmup_steps = min(distance_to_end, self.num_microbatches)
        # The remaining microbatches will be processed in the steady state
        num_steady_steps = self.num_microbatches - num_warmup_steps

        # Reset state for the new global batch
        self.forward_logits = []
        self.total_loss = 0.0

        # --- PHASE 1: WARM-UP ---
        # Run forward steps to fill the pipeline.
        for i in range(num_warmup_steps):
            self._run_forward_step(x_microbatches, i)

        # --- PHASE 2: 1F1B STEADY STATE ---
        # Run ONE forward step, immediately followed by ONE backward step.
        self.total_loss = 0.0
        for i in range(0, num_steady_steps):
            # Forward processes the next available micro-batch
            self._run_forward_step(x_microbatches, i + num_warmup_steps)
            # Backward processes the oldest micro-batch
            self._run_backward_step(target_microbatches, i)

        # --- PHASE 3: COOLDOWN ---
        # Run backward steps to drain the remaining activations in the queue.
        for i in range(num_steady_steps, self.num_microbatches):
            self._run_backward_step(target_microbatches, i)

        if self.stage.is_last_stage:
            self.total_loss /= self.num_microbatches
            return self.total_loss

        return None
