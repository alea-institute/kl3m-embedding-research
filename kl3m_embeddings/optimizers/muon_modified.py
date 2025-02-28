"""
Forked version of Muon with small torch and precision changes.

Based on the original MIT Muon implementation by Jordan Keller:
https://github.com/KellerJordan/Muon
"""

# imports
import os
from typing import List, Dict, Any, Optional, Tuple

# packages
import torch
import torch.distributed as dist


@torch.compile
def compute_orthogonalization(
    matrix: torch.Tensor, iteration_steps: int = 10, epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Computes the zeroth power / orthogonalization of a matrix using Newton-Schulz iteration.

    This implementation uses a quintic iteration with coefficients selected to maximize the slope
    at zero. The iteration produces a matrix close to UV^T where USV^T is the SVD of the input
    matrix. The output's singular values are approximately uniformly distributed in [0.5, 1.5].

    Args:
        matrix: Input matrix to orthogonalize (must be 2D)
        iteration_steps: Number of Newton-Schulz iterations to perform
        epsilon: Small constant for numerical stability

    Returns:
        Orthogonalized matrix of the same shape as input

    Raises:
        AssertionError: If input matrix is not 2D
    """
    assert len(matrix.shape) == 2, "Input matrix must be 2D"

    # Coefficients for the quintic iteration
    alpha, beta, gamma = (3.4445, -4.7750, 2.0315)

    # Normalize matrix to ensure top singular value <= 1
    matrix /= matrix.norm() + epsilon

    # Transpose if matrix is tall (more rows than columns)
    is_tall_matrix = matrix.size(0) > matrix.size(1)
    if is_tall_matrix:
        matrix = matrix.T

    # Perform Newton-Schulz iterations
    for _ in range(iteration_steps):
        # Compute intermediate matrices for the quintic iteration
        product_matrix = matrix @ matrix.T
        combined_matrix = beta * product_matrix + gamma * (
            product_matrix @ product_matrix
        )
        matrix = alpha * matrix + combined_matrix @ matrix

    # Restore original matrix orientation if transposed
    if is_tall_matrix:
        matrix = matrix.T

    return matrix


class Muon(torch.optim.Optimizer):
    """
    Muon (MomentUm Orthogonalized by Newton-schulz) Optimizer

    This optimizer combines SGD-momentum with an orthogonalization post-processing step.
    For 2D parameters, each update is replaced with the nearest orthogonal matrix using
    an efficient Newton-Schulz iteration that can run stably in bfloat16 on GPU.

    The optimizer automatically identifies which parameters should use Muon optimization
    versus standard AdamW optimization based on tensor dimensionality and size.

    Important Notes:
        - Best suited for large batch size training
        - May not be optimal for finetuning pretrained models
        - Automatically handles distributed training environments

    Args:
        muon_params: Parameters to optimize with Muon
        lr: Learning rate (spectral norm of updates)
        momentum: Momentum factor for internal SGD
        nesterov: Whether to use Nesterov momentum
        ns_steps: Number of Newton-Schulz iterations
        adamw_params: Parameters to optimize with AdamW (optional)
        adamw_lr: Learning rate for AdamW parameters
        adamw_betas: Beta parameters for AdamW (momentum and variance)
        adamw_eps: Epsilon parameter for AdamW numerical stability
        adamw_wd: Weight decay for AdamW parameters
    """

    def __init__(
        self,
        muon_params: List[torch.Tensor | torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        adamw_params: Optional[List[torch.Tensor | torch.nn.Parameter]] = None,
        adamw_lr: float = 3e-4,
        adamw_betas: Tuple[float, float] = (0.95, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0,
    ):
        # Initialize optimizer defaults
        defaults: Dict[str, Any] = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adamw_lr_ratio": adamw_lr / lr,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
            "adamw_wd": adamw_wd,
        }

        # Combine all parameters
        all_params: List[torch.Tensor] = list(muon_params)
        if adamw_params is not None:
            all_params.extend(list(adamw_params))

        super().__init__(all_params, defaults)

        # determine precision from params
        self.precision = (
            torch.bfloat16
            if any(p.dtype == torch.bfloat16 for p in all_params)
            else torch.float32
        )

        # Determine which parameters should use Muon optimization
        for param in muon_params:
            # Use Muon for parameters that are:
            # 1. At least 2D (matrices or higher)
            # 2. Not too large (to avoid embedding/head layers)
            should_use_muon = param.ndim >= 2 and param.size(0) < 10000
            self.state[param]["use_muon"] = should_use_muon

        # Mark AdamW parameters explicitly
        if adamw_params is not None:
            for param in adamw_params:
                self.state[param]["use_muon"] = False

        # Set up distributed training parameters
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))

    def step(self) -> None:
        """
        Performs a single optimization step.

        This method:
        1. Applies Muon optimization to eligible parameters
        2. Applies AdamW optimization to remaining parameters
        3. Handles distributed training synchronization
        """
        for group in self.param_groups:
            self._apply_muon_updates(group)
            self._apply_adamw_updates(group)

    def _apply_muon_updates(self, group: Dict[str, Any]) -> None:
        """
        Applies Muon optimization updates to eligible parameters.

        Args:
            group: Parameter group containing optimization settings
        """
        # Get Muon-eligible parameters
        muon_params = [p for p in group["params"] if self.state[p]["use_muon"]]
        learning_rate = group["lr"]
        momentum_factor = group["momentum"]

        # Initialize flat tensor for distributed updates
        total_params = sum(p.numel() for p in muon_params)
        flat_updates = torch.zeros(total_params, device="cuda", dtype=self.precision)

        # Generate updates for parameters assigned to this process
        current_index = 0
        for param_idx, param in enumerate(muon_params):
            # check if we require grad and not none
            if param.grad is not None and param.requires_grad:
                gradient = param.grad
                if gradient.ndim > 2:
                    gradient = gradient.view(gradient.size(0), -1)

                # Initialize or update momentum buffer
                param_state = self.state[param]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.zeros_like(gradient)
                momentum_buffer = param_state["momentum_buffer"]

                # Apply momentum
                momentum_buffer.mul_(momentum_factor).add_(gradient)
                if group["nesterov"]:
                    gradient = gradient.add(momentum_buffer, alpha=momentum_factor)

                # Compute orthogonalized update
                orthogonalized_gradient = compute_orthogonalization(
                    gradient, iteration_steps=group["ns_steps"]
                )

                # Scale gradient based on matrix shape
                shape_scale = max(1, gradient.size(0) / gradient.size(1)) ** 0.5
                orthogonalized_gradient *= shape_scale

                # Store in flat tensor
                flat_updates[current_index : current_index + param.numel()] = (
                    orthogonalized_gradient.flatten()
                )

            current_index += param.numel()

        # Synchronize updates across devices in distributed setting
        if self.world_size > 1:
            dist.all_reduce(flat_updates, op=dist.ReduceOp.SUM)

        # Apply updates to parameters
        current_index = 0
        for param in muon_params:
            grad_update = flat_updates[
                current_index : current_index + param.numel()
            ].view_as(param.data)
            param.data.add_(grad_update.type_as(param.data), alpha=-learning_rate)
            current_index += param.numel()

    def _apply_adamw_updates(self, group: Dict[str, Any]) -> None:
        """
        Applies AdamW optimization updates to non-Muon parameters.

        Args:
            group: Parameter group containing optimization settings
        """
        # Get parameters for AdamW update
        adamw_params = [p for p in group["params"] if not self.state[p]["use_muon"]]

        # Extract AdamW hyperparameters
        learning_rate = group["adamw_lr_ratio"] * group["lr"]
        beta1, beta2 = group["adamw_betas"]
        epsilon = group["adamw_eps"]
        weight_decay = group["adamw_wd"]

        for param in adamw_params:
            if param.grad is not None and param.requires_grad:
                gradient = param.grad
                assert gradient is not None, "Gradient cannot be None for AdamW update"

                # Initialize or get parameter state
                param_state = self.state[param]
                if "step" not in param_state:
                    param_state.update(
                        {
                            "step": 0,
                            "moment1": torch.zeros_like(gradient),
                            "moment2": torch.zeros_like(gradient),
                        }
                    )

                # Update step count and get buffers
                param_state["step"] += 1
                step_count = param_state["step"]
                moment1_buffer = param_state["moment1"]
                moment2_buffer = param_state["moment2"]

                # Update moment estimates
                moment1_buffer.lerp_(gradient, 1 - beta1)
                moment2_buffer.lerp_(
                    gradient.square(), 1 - beta2
                )  # NOTE: forcing .bfloat16() as .square() is float

                # Compute bias-corrected update
                bias_correction1 = 1 - beta1**step_count
                bias_correction2 = 1 - beta2**step_count
                update_scale = bias_correction1 / bias_correction2**0.5

                # Compute effective gradient
                effective_gradient = moment1_buffer / (epsilon + moment2_buffer.sqrt())

                # Apply weight decay and update
                param.data.mul_(1 - learning_rate * weight_decay)
                param.data.add_(effective_gradient, alpha=-learning_rate / update_scale)
