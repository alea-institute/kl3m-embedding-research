"""
Exponential warmup cooldown scheduler.
"""

# imports
from typing import Optional, Union

# packages
import torch


class ExponentialWarmupCooldownScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Generalized exponential scheduler inspired by that increases and decreases by multiplicative factors
    for M periods of K steps per phase.

    The scheduler:
    1. increases lr *= factor for each of M periods of A steps; the lr is constant within each period
    2. holds lr constant for B steps;
    3. decreases lr /= factor for each of N periods of C steps; the lr is constant within each period
    """

    # pylint: disable=too-many-positional-arguments,too-many-arguments
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        start_lr: float = 1e-6,
        end_lr: float = 1e-7,
        warmup_factor: float = 2.0,
        warmup_steps_per_period: int = 10000,
        warmup_periods: int = 8,
        constant_steps: int = 100000,
        cooldown_factor: float = 2.0,
        cooldown_steps_per_period: int = 10000,
        cooldown_periods: int = 8,
        last_epoch: int = -1,
    ):
        """
        Initialize the ExponentialWarmupCooldownScheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.
            start_lr (float): The start learning rate.
            end_lr (float): The end learning rate.
            warmup_factor (float): The warmup factor.
            warmup_steps_per_period (int): The number of warmup steps per period.
            warmup_periods (int): The number of warmup periods.
            constant_steps (int): The number of constant steps.
            cooldown_factor (float): The cooldown factor.
            cooldown_steps_per_period (int): The number of cooldown steps per period.
            cooldown_periods (int): The number of cooldown periods.
            last_epoch (int): The last epoch.

        Note:
            The total number of steps is equal to the sum of the steps in each phase.
        """
        # initialize the scheduler
        self.warmup_factor = warmup_factor
        self.warmup_steps_per_period = warmup_steps_per_period
        self.warmup_periods = warmup_periods
        self.constant_steps = constant_steps
        self.cooldown_factor = cooldown_factor
        self.cooldown_steps_per_period = cooldown_steps_per_period
        self.cooldown_periods = cooldown_periods
        self.current_step = 0
        self.lr = start_lr
        self.end_lr = end_lr

        # calculate phase step counts
        self.steps_in_phase = 0
        self.periods_in_phase = 0
        self.total_warmup_steps = self.warmup_steps_per_period * self.warmup_periods
        self.total_constant_steps = self.constant_steps
        self.total_cooldown_steps = (
            self.cooldown_steps_per_period * self.cooldown_periods
        )

        # call the parent
        super().__init__(optimizer, self.get_lr, last_epoch=last_epoch)

    def get_current_phase(self) -> str:
        """
        Get the current phase.

        Returns:
            str: The current phase.
        """
        if self.current_step < self.total_warmup_steps:
            return "warmup"
        if self.current_step < self.total_warmup_steps + self.total_constant_steps:
            return "constant"
        return "cooldown"

    # pylint: disable=unused-argument
    def get_lr(self, epoch: Optional[int] = None) -> Union[float, list[float]]:
        """
        Get the learning rate lambda.

        Returns:
            float: The learning rate.
        """
        # set the current step
        self.current_step += 1

        # check phase and step
        phase = self.get_current_phase()
        lr = self.lr
        if phase == "warmup" and self.steps_in_phase >= self.warmup_steps_per_period:
            lr *= self.warmup_factor
            self.steps_in_phase = 0
            self.periods_in_phase += 1
        elif phase == "constant" and self.steps_in_phase >= self.constant_steps:
            self.steps_in_phase = 0
            self.periods_in_phase += 1
        elif (
            phase == "cooldown"
            and self.steps_in_phase >= self.cooldown_steps_per_period
        ):
            lr = max(lr / self.cooldown_factor, self.end_lr)
            self.steps_in_phase = 0
            self.periods_in_phase += 1
        else:
            self.steps_in_phase += 1

        # set the last lr
        self.lr = lr

        # return the lr
        return [lr] * len(self.optimizer.param_groups)

    def __str__(self) -> str:
        """
        Get the string representation.

        Returns:
            str: The string representation.
        """
        return f"ExponentialWarmupCooldownScheduler(lr={self.lr}, step={self.current_step})"
