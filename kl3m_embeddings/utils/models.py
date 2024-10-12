"""
Utilities for working with models.
"""

# imports

# packages
import torch


def get_model_size(model: torch.nn.Module, grad_only: bool = True) -> int:
    """
    Get the size of a model in bytes.

    Args:
        model (torch.nn.Module): model to get size of
        grad_only (bool): whether to only count gradients

    Returns:
        int: size of model in bytes
    """
    if grad_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def get_model_size_str(model: torch.nn.Module, grad_only: bool = True) -> str:
    """
    Get the size of a model as a human-readable string.

    Args:
        model (torch.nn.Module): model to get size of
        grad_only (bool): whether to only count gradients

    Returns:
        str: size of model as human-readable string
    """
    size = get_model_size(model, grad_only)
    if size > 1e9:
        return f"{size / 1e9:.2f}B"

    if size > 1e6:
        return f"{(size / 1e6):0.1f}M"

    if size > 1e3:
        return f"{(size / 1e3):0.1f}K"

    return f"{size}B"
