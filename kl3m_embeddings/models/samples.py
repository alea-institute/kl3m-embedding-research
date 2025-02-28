"""
Sampling client for embedding models.
"""

# imports
import logging
import time
from collections import Counter
from random import choices
from typing import Any, Optional

# packages
import httpx
import torch

# project


# pylint: disable=too-many-positional-arguments
def get_embedding_sample(
    task_probabilities: dict[str, float],
    logger: logging.Logger,
    batch_size: int = 1,
    endpoint_url: str = "http://localhost:8000/",
    endpoint_client: Optional[httpx.Client] = None,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Get a sample for training and related metadata/stats.

    Args:
        task_probabilities (dict[str, float]): The task probabilities.
        batch_size (int): The batch size.
        endpoint_url (str): The endpoint URL.
        endpoint_client (httpx.Client): The endpoint client.
        logger (logging.Logger): The logger.
        device (str): The device to use.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, Any]]: The sample and stats.
    """
    # check if we have task probabilities
    if len(task_probabilities) == 0:
        raise ValueError("No task probabilities provided")

    # get a task based on probability dist
    tasks, task_weights = zip(*task_probabilities.items())
    task = choices(
        population=tasks,
        weights=task_weights,
        k=1,
    ).pop()

    # get url and request an MLM batch
    batch_url = f"{endpoint_url.rstrip('/')}/batch/{task}"
    batch_post_body = {
        "batch_size": batch_size,
    }

    # create a client if we don't have one
    if endpoint_client is None:
        endpoint_client = httpx.Client(http2=True)

    # post request and try again until we get something
    response = None
    while True:
        try:
            response = endpoint_client.post(
                batch_url, json=batch_post_body, timeout=10.0
            )

            if response.status_code == 200:
                break

            if response.status_code in (503,):
                # sleep and try again in 1 second without overwhelming the server
                logger.warning("Server busy, retrying...")
                time.sleep(1)
                continue

            # otherwise, raise an error
            response.raise_for_status()
        except (
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
        ):
            logger.warning("Timeout on request, retrying...")
            continue
        except Exception as e:
            logger.error("Error on request: %s", str(e))
            raise e

    # fail if we got here somehow
    if response is None:
        raise ValueError("No response from server")

    # parse and convert to torch tensors on device
    data = response.json()

    # convert data to tensor dict and return
    result = {
        "input_ids": torch.tensor(
            [r["input_ids"] for r in data],
            device=device,
        ),
        "attention_mask": torch.tensor(
            [r["attention_mask"] for r in data],
            device=device,
        ),
        "labels": torch.tensor(
            [r["labels"] for r in data],
            device=device,
        ),
    }

    if "token_type_ids" in data[0]:
        result["token_type_ids"] = torch.tensor(
            [r["token_type_ids"] for r in data],
            device=device,
        )

    # count tokens by dataset
    tokens_by_dataset: Counter[str] = Counter()
    for r in data:
        for dataset_id in r.get("dataset_id", []):
            tokens_by_dataset[dataset_id] += sum(r["attention_mask"])

    # flatten identifiers, dataset_ids, sources, and mime_types
    flat_identifiers = []
    flat_dataset_ids = []
    flat_sources = []
    flat_mime_types = []

    for r in data:
        flat_identifiers.extend(r["identifier"])
        flat_dataset_ids.extend(r["dataset_id"])
        flat_sources.extend(r["source"])
        flat_mime_types.extend(r["mime_type"])

    # get sample metadata
    sample_metadata = {
        "task": task,
        "num_samples": len(data),
        "num_identifiers": len(set(flat_identifiers)),
        "num_tokens": sum(sum(r["attention_mask"]) for r in data),
        "samples_by_dataset": Counter(flat_dataset_ids),
        "tokens_by_dataset": tokens_by_dataset,
        "objects": list(set(flat_identifiers)),
    }

    return result, sample_metadata


def get_random_source(
    endpoint_url: str = "http://localhost:8000/",
    endpoint_client: Optional[httpx.Client] = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Get a random source record from /source/random for the model to use in
    contrastive training.

    Args:
        endpoint_url (str): The endpoint URL.
        endpoint_client (httpx.Client): The endpoint client.
        device (str): The device to use.

    Returns:
        dict[str, torch.Tensor]: The sample.
    """
    # get url and request an MLM batch
    batch_url = f"{endpoint_url.rstrip('/')}/source/random"

    # create a client if we don't have one
    if endpoint_client is None:
        endpoint_client = httpx.Client(http2=True)

    # post request and try again until we get something
    response = None
    while True:
        try:
            response = endpoint_client.get(batch_url, timeout=10.0)

            if response.status_code == 200:
                break

            if response.status_code in (503,):
                # sleep and try again in 1 second without overwhelming the server
                time.sleep(1)
                continue

            # otherwise, raise an error
            response.raise_for_status()
        except (
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
        ):
            continue
        except Exception as e:
            raise e

    # fail if we got here somehow
    if response is None:
        raise ValueError("No response from server")

    # parse and convert to torch tensors on device
    data = response.json()

    return data
