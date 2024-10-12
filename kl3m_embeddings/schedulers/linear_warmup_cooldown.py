"""
Linear warmup cooldown scheduler (generalized trapezoidal scheduling).
"""

# imports
from typing import Optional, Union

# packages
import torch


class LinearWarmupCooldownScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Generalized linear scheduled inspired by trapezoidal scheduling; if start and end lr are identical, then it is
    equivalent to trapezoidal scheduling.  If start and end lr are not provided, then they are set asymmetrically to
    1/100 and 1/1000 of the peak learning rate, respectively.

    The scheduler:
    1. increases from start_lr to peak_lr in warmup_steps linearly by step
    2. decreases from peak_lr to end_lr in total_steps - warmup_steps
    3. remains constant at end_lr until training is terminated
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr=1e-4,
        start_lr: Optional[float] = None,
        end_lr: Optional[float] = None,
        warmup_steps: int = 1000,
        peak_steps: int = 990000,
        total_steps: int = 2000000,
        last_epoch: int = -1,
    ):
        """
        Initialize the LinearWarmupScheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.
            peak_lr (float): The learning rate.
            start_lr (float): The start learning rate; defaults to 1/100 of lr.
            end_lr (float): The end learning rate; defaults to 1/1000 of lr.
            warmup_steps (int): The number of warmup steps.
            peak_steps (int): The number of peak steps.
            total_steps (int): The total number of steps.
            last_epoch (int): The last epoch.
        """
        # initialize the scheduler
        self.peak_lr = peak_lr
        self.current_step = 0
        self.start_lr = start_lr or peak_lr / 100
        self.end_lr = end_lr or peak_lr / 1000
        self.warmup_steps = warmup_steps
        self.peak_steps = peak_steps
        self.total_steps = total_steps
        self.lr = start_lr

        # call the parent
        super().__init__(optimizer, self.get_lr, last_epoch=last_epoch)

    # pylint: disable=unused-argument
    def get_lr(self, epoch: Optional[int] = None) -> Union[float, list[float]]:
        """
        Get the learning rate lambda.

        Args:
            epoch (int): The epoch.

        Returns:
            float: The learning rate.
        """
        # set the current step
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            #  linearly warm up from [0, warmup_steps] from [start_lr, peak_lr]
            lr = (
                self.start_lr
                + (self.peak_lr - self.start_lr) * self.current_step / self.warmup_steps
            )
        elif self.warmup_steps <= self.current_step < self.peak_steps:
            # peak_lr from [warmup_steps, warmup_steps + peak_steps]
            lr = self.peak_lr
        elif self.peak_steps <= self.current_step < self.total_steps:
            # linearly cool down from [warmup_steps + peak_steps, total_steps]
            lr = self.peak_lr - (self.peak_lr - self.end_lr) * (
                self.current_step - self.peak_steps
            ) / (self.total_steps - self.peak_steps)
        else:
            # end_lr if we are still going
            lr = self.end_lr

        # inc and return
        return [lr for _ in self.optimizer.param_groups]
