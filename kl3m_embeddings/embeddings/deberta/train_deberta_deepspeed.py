"""
Train a kl3m deberta model with given configuration using deepspeed, which supports
both distributed training and various optimization and parallelization strategies.

Note that these models are trained with Matroyshka-style dimensionality reduction, but
they can be used with the standard DebertaV2 huggingface architecture on the normal
.forward() after training without any issues.
"""

# imports
import argparse
from collections import Counter
from math import log2
from pathlib import Path
from random import randint, random
from typing import Optional

# packages
import httpx
import torch
from transformers import DebertaV2Config, PreTrainedTokenizerFast

# project
from kl3m_embeddings.embeddings.deberta.matroyshka_deberta import (
    MatroyshkaDebertaV2ForMaskedLM,
)
from kl3m_embeddings.embeddings.deepspeed_trainer import KL3MDeepspeedTrainer
from kl3m_embeddings.utils.models import get_model_size_str

# constants
DEFAULT_TOKENIZER = "alea-institute/kl3m-003-64k"
DEFAULT_ENDPOINT = "http://localhost:8000"


class KL3MDeepspeedDebertaTrainer(KL3MDeepspeedTrainer):
    """
    Trainer for kl3m deberta model.
    """

    def __init__(
        self,
        config_path: Path,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        checkpoint_path: Path = Path("checkpoint/"),
    ):
        # call the super
        super().__init__(
            config_path=config_path,
            tokenizer_name=tokenizer_name,
            checkpoint_path=checkpoint_path,
        )

        # get the matroyshka sampling probability
        self.matroyshka_probability = self.training_config.get(
            "matroyshka_probability", 0.5
        )

        # get the endpoint url with client
        self.endpoint_url = self.training_config.get("endpoint_url", DEFAULT_ENDPOINT)
        self.endpoint_client = httpx.Client(
            http2=True,
            limits=httpx.Limits(
                max_keepalive_connections=4, max_connections=16, keepalive_expiry=10
            ),
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close the endpoint client
        try:
            self.endpoint_client.close()
        except Exception as e:  # pylint: disable=broad-except
            self.log(
                "Error closing endpoint client during cleanup: %s",
                str(e),
                level="error",
            )

    def setup_model(
        self,
        tokenizer: PreTrainedTokenizerFast,
        precision: Optional[torch.dtype] = None,
    ) -> None:
        """
        Get the model.

        Args:
            tokenizer (PreTrainedTokenizerFast): The tokenizer.
            precision (torch.dtype): The precision.
        """
        # get default precision if not passed
        precision = precision or self.precision

        # try to load from the checkpoint path
        if self.checkpoint_path.exists():
            try:
                self.model = MatroyshkaDebertaV2ForMaskedLM.from_pretrained(
                    self.checkpoint_path,
                )
                self.model.train()
                self.log("Loaded model from checkpoint.")
            except Exception as e:  # pylint: disable=broad-except
                self.log(
                    "Error loading model from checkpoint: %s", str(e), level="error"
                )

        if self.model is None:
            # load from config path
            model_config = DebertaV2Config.from_pretrained(self.config_path)

            # set the vocab size and max position embeddings
            model_config.vocab_size = len(self.tokenizer)
            self.model = MatroyshkaDebertaV2ForMaskedLM(model_config)
            self.model.train()

            self.log("Loaded model from config.")

    def get_sample(self, device: Optional[str] = "cpu") -> dict[str, torch.Tensor]:
        """
        Get a sample for training.

        Returns:
            dict[str, torch.Tensor]: The sample.
        """
        # get url and request
        mlm_url = f"{self.endpoint_url.rstrip('/')}/batch/mlm"
        mlm_post_body = {
            "batch_size": self.training_config.get("batch_size", 1),
        }

        # post request and try again until we get something
        response = None
        while True:
            try:
                response = self.endpoint_client.post(
                    mlm_url, json=mlm_post_body, timeout=1.0
                )
                if response.status_code == 200:
                    break

                if response.status_code not in (503,):
                    response.raise_for_status()
            except (
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ):
                self.log("Timeout on request, retrying...", level="warning")
                continue
            except Exception as e:
                self.log("Error on request: %s", str(e), level="error")
                raise e

        # fail if we got here somehow
        if response is None:
            raise ValueError("No response from server")

        # parse and convert to torch tensors on device
        data = response.json()

        # count the datasets and add them to the current entry
        dataset_list = [r["dataset_id"] for r in data]
        dataset_count = Counter(dataset_list)
        self.step_entry["datasets"] = dataset_count

        # update the current log entry
        object_list = [r["identifier"] for r in data]
        self.step_entry["objects"] = object_list

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

        # get matroyshka samples
        if random() < self.matroyshka_probability:
            max_log2 = int(log2(self.model.config.hidden_size)) - 1  # type: ignore
            result["reduced_dim"] = 2 ** randint(1, max_log2)
            self.step_entry["reduced_dim"] = result["reduced_dim"]

        return result


if __name__ == "__main__":
    # set up args
    parser = argparse.ArgumentParser(
        description="Train a deberta model with given configuration."
    )
    parser.add_argument(
        "config_path", type=Path, help="Path to the model configuration"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help="Name of the tokenizer",
        default=DEFAULT_TOKENIZER,
    )
    parser.add_argument(
        "--checkpoint_path", type=Path, help="Path to the checkpoint file", default=None
    )
    args = parser.parse_args()

    # create the trainer
    trainer = KL3MDeepspeedDebertaTrainer(
        config_path=args.config_path,
        tokenizer_name=args.tokenizer_name,
        checkpoint_path=args.checkpoint_path,
    )

    # print key tokenizer and model info
    print(f"Tokenizer Size: {len(trainer.tokenizer)}")
    print(f"Model Size: {get_model_size_str(trainer.model)}")
    print(f"Training Precision: {trainer.precision}")

    # train the model
    trainer.train()
