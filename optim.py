from typing import Optional

import torch

# Slanted Triangular Learning Rate Scheduler
class STLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            steps_per_epoch: Optional[int] = None,
            cut_frac: float = 0.1,
            ratio: int = 32,
            last_epoch: int = -1,
    ):
        self.num_epochs = epochs
        self.num_steps_per_epoch = steps_per_epoch
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.total_steps = epochs if steps_per_epoch is None else epochs * steps_per_epoch
        self.cut = int(self.total_steps * self.cut_frac)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        prop = step / self.cut if step < self.cut else 1 - (step - self.cut) / (self.total_steps - self.cut)
        multiplier = (1 + prop * (self.ratio - 1)) / self.ratio
        return [lr * multiplier for lr in self.base_lrs]
