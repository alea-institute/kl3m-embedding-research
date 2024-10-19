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
        peak_lr: float = 1e-4,
        start_lr: Optional[float] = None,
        end_lr: Optional[float] = None,
        warmup_steps: int = 100,
        peak_steps: int = 900,
        total_steps: int = 200,
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
        self.start_lr = start_lr or peak_lr / 100
        self.end_lr = end_lr or peak_lr / 1000
        self.warmup_steps = warmup_steps
        self.peak_steps = peak_steps
        self.total_steps = total_steps
        self.cooldown_steps = total_steps - warmup_steps - peak_steps

        # get state values
        self.current_step = 0
        self.current_lr = start_lr

        # call the parent
        super().__init__(optimizer, self.get_lr, last_epoch=last_epoch)

    def get_current_phase(self) -> str:
        """
        Get the current phase.

        Returns:
            str: The phase.
        """
        if self.current_step < self.warmup_steps:
            return "warmup"

        if self.warmup_steps <= self.current_step < self.warmup_steps + self.peak_steps:
            return "peak"

        return "cooldown"

    # pylint: disable=unused-argument
    def get_lr(self, epoch: Optional[int] = None) -> Union[float, list[float]]:
        """
        Get the learning rate lambda.

        Args:
            epoch (int): The epoch.

        Returns:
            float: The learning rate.
        """
        # get the phase
        phase = self.get_current_phase()

        # set the learning rate
        if phase == "warmup":
            warmup_pct = self.current_step / self.warmup_steps
            self.current_lr = self.start_lr + warmup_pct * (
                self.peak_lr - self.start_lr
            )
        elif phase == "peak":
            self.current_lr = self.peak_lr
        else:
            cooldown_pct = (
                self.current_step - self.warmup_steps - self.peak_steps
            ) / self.cooldown_steps
            self.current_lr = max(
                self.peak_lr - cooldown_pct * (self.peak_lr - self.end_lr), self.end_lr
            )

        # return the learning rate
        return [self.current_lr] * len(self.optimizer.param_groups)

    def __str__(self) -> str:
        """
        Get the string representation.

        Returns:
            str: The string representation.
        """
        return f"LinearWarmupCooldownScheduler(lr={self.current_lr}, step={self.current_step})"


if __name__ == "__main__":
    # test the scheduler
    optimizer = torch.optim.Adam([torch.zeros(1)], lr=1e-4)
    scheduler = LinearWarmupCooldownScheduler(
        optimizer,
        start_lr=0.0001,
        peak_lr=0.0003,
        warmup_steps=10000,
        peak_steps=90000,
        total_steps=200000,
    )
    for step in range(0, 210000, 1000):
        print(step, scheduler.get_lr())
        if scheduler.current_step == 2000000:
            break
