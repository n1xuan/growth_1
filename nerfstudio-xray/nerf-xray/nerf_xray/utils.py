# Based on nerfstudio.engine.schedulers
"""Various utilities for training"""

from dataclasses import dataclass, field
from typing import Optional, Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler

from torch.optim.lr_scheduler import LRScheduler
from nerfstudio.engine.schedulers import Scheduler, SchedulerConfig

@dataclass
class ColdRestartLinearDecaySchedulerConfig(SchedulerConfig):
    """Config for linear decay scheduler with warmup"""

    _target: Type = field(default_factory=lambda: ColdRestartLinearDecayScheduler)
    """target class to instantiate"""
    lr_pre_warmup: float = 1e-8
    """Learning rate before warmup."""
    lr_final: Optional[float] = None
    """Final learning rate. If not provided, it will be set to the optimizers learning rate."""
    warmup_steps: int = 0
    """Number of warmup steps."""
    steady_steps: int = 10000
    """Number of steady steps."""
    max_steps: int = 100000
    """The maximum number of steps. From steady_steps to max_steps, the learning rate will decay to lr_final."""

class ColdRestartLinearDecayScheduler(Scheduler):
    """Linear decay scheduler with warmup.
    First, the learning rate will be increased from lr_pre_warmup to lr_init using a sine function.
    Then, the learning rate will be kept constant for until steady_steps.
    Finally, the learning rate will be linearly decayed to lr_final until max_steps.
    """

    config: ColdRestartLinearDecaySchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        if self.config.lr_final is None:
            lr_final = lr_init
        else:
            lr_final = self.config.lr_final

        def func(step):
            if step < self.config.warmup_steps:
                lr = self.config.lr_pre_warmup + (lr_init - self.config.lr_pre_warmup) * np.sin(
                    0.5 * np.pi * np.clip(step / self.config.warmup_steps, 0, 1)
                )
            elif step < self.config.steady_steps:
                lr = lr_init
            elif step < self.config.max_steps:
                lr = lr_init + (lr_final - lr_init) * np.clip((step - self.config.steady_steps) / (self.config.max_steps - self.config.steady_steps), 0, 1)
            else:
                lr = lr_final
            return lr / lr_init  # divided by lr_init because the multiplier is with the initial learning rate

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler

