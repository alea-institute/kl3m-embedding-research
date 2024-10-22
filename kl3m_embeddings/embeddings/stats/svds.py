"""
SVD/spectral-related layer metrics for a model.
"""

# imports
from typing import Any

# packages
import numpy
import torch


def get_layer_svd_stats(layer_tensor: torch.Tensor) -> dict[str, Any]:
    """
    Get the SVD and other statistics for a given layer tensor.

    Args:
        layer_tensor (torch.Tensor): The layer tensor.

    Returns:
        dict[str, Any]: The statistics for the layer tensor.
    """
    layer_stats = {
        "svd_values": None,
        "parameters": layer_tensor.numel(),
        "norm": layer_tensor.norm().item(),
        "mean": layer_tensor.mean().item(),
        "min": layer_tensor.min().item(),
        "max": layer_tensor.max().item(),
        "scree_1": None,
        "scree_2": None,
        "mean_ratio_1": None,
        "median_ratio_1": None,
    }

    # get the norm, mean, min, and max

    # compute the SVD with torch.linalg.svdvals()
    if layer_tensor.device.type == "cuda":
        print("Using gesvda for CUDA SVD.")
        layer_svd_values = (
            torch.linalg.svdvals(layer_tensor, driver="gesvda").cpu().numpy()  # pylint: disable=not-callable
        )
    else:
        # make sure dtype is float32
        layer_svd_values = torch.linalg.svdvals(layer_tensor.float()).numpy()  # pylint: disable=not-callable

    # store the SVD values in the dictionary
    layer_stats["svd_values"] = layer_svd_values.tolist()

    # scree 1 and 2 are the v1 and v1+v2 / sum of all values
    svd_sum = layer_svd_values.sum()
    svd_mean = layer_svd_values.mean()
    svd_median = numpy.nanmedian(layer_svd_values)
    layer_stats["scree_1"] = layer_svd_values[0] / svd_sum
    layer_stats["scree_2"] = (layer_svd_values[0] + layer_svd_values[1]) / svd_sum
    layer_stats["mean_ratio_1"] = layer_svd_values[0] / svd_mean
    layer_stats["median_ratio_1"] = layer_svd_values[0] / svd_median

    return layer_stats


def get_all_layer_svd_stats(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """
    Get the SVDs for all 2D layers of a model by
    iterating through the state dict, checking the shape dimension,
    and then computing the SVD for the layer if it is 2D.

    Args:
        state_dict (dict[str, torch.Tensor]): The state dict of the model.

    Returns:
        dict[str, Any]: The SVDs for all 2D layers of the model.
    """
    # initialize the svd dictionary by layer
    layer_metrics = {}

    # iterate through the state dict
    for layer_name, layer_data in state_dict.items():
        # check if the value is a 2D tensor
        try:
            if len(layer_data.shape) == 2:
                # get layer stats
                layer_metrics[layer_name] = get_layer_svd_stats(layer_data)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error processing layer {layer_name}: {e}")

    #  now return the layer metrics
    return layer_metrics


def get_aggregate_model_stats(layer_metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Get the aggregate statistics for the model from the layer metrics.

    Args:
        layer_metrics (dict[str, Any]): The layer metrics.

    Returns:
        dict[str, Any]: The aggregate model statistics.
    """
    # now get all aggregate metrics
    total_params = sum(
        layer_metrics[layer_name]["parameters"] for layer_name in layer_metrics
    )
    metrics = {
        # "layers": layer_metrics,
        "scree_1": numpy.nanmean(
            [layer_metrics[layer_name]["scree_1"] for layer_name in layer_metrics]
        ),
        "scree_2": numpy.nanmean(
            [layer_metrics[layer_name]["scree_2"] for layer_name in layer_metrics]
        ),
        "mean_ratio_1": numpy.nanmean(
            [layer_metrics[layer_name]["mean_ratio_1"] for layer_name in layer_metrics]
        ),
        "median_ratio_1": numpy.nanmean(
            [
                layer_metrics[layer_name]["median_ratio_1"]
                for layer_name in layer_metrics
            ]
        ),
        # weighted by num params
        "scree_1_weighted": numpy.nanmean(
            [
                layer_metrics[layer_name]["scree_1"]
                * layer_metrics[layer_name]["parameters"]
                for layer_name in layer_metrics
            ]
        )
        / total_params,
        "scree_2_weighted": numpy.nanmean(
            [
                layer_metrics[layer_name]["scree_2"]
                * layer_metrics[layer_name]["parameters"]
                for layer_name in layer_metrics
            ]
        )
        / total_params,
        "mean_ratio_1_weighted": numpy.nanmean(
            [
                layer_metrics[layer_name]["mean_ratio_1"]
                * layer_metrics[layer_name]["parameters"]
                for layer_name in layer_metrics
            ]
        )
        / total_params,
        "median_ratio_1_weighted": numpy.nanmean(
            [
                layer_metrics[layer_name]["median_ratio_1"]
                * layer_metrics[layer_name]["parameters"]
                for layer_name in layer_metrics
            ]
        )
        / total_params,
    }

    # add mean_ratio_1for all layers with their key
    for layer_name in layer_metrics:
        metrics[layer_name + "_mean_ratio_1"] = layer_metrics[layer_name][
            "mean_ratio_1"
        ]

    return metrics


def get_model_svd_stats(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """
    Get the SVDs for all 2D layers of a model by
    iterating through the state dict, checking the shape dimension,
    and then computing the SVD for the layer if it is 2D.

    Args:
        state_dict (dict[str, torch.Tensor]): The state dict of the model.

    Returns:
        dict[str, Any]: The SVDs for all 2D layers of the model.
    """
    # initialize the svd dictionary by layer
    layer_metrics = {}

    # get all layer metrics
    layer_metrics = get_all_layer_svd_stats(state_dict)

    # get the aggregate model stats
    model_metrics = get_aggregate_model_stats(layer_metrics)

    # return the model metrics
    return model_metrics
