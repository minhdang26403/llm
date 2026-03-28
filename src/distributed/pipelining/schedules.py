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
