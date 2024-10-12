"""
Train a kl3m deberta model with given configuration on a single node, single GPU setup
with pure torch.  transformers is only used to load the model architecture and tokenizer.

Note that these models are trained with Matroyshka-style dimensionality reduction, but
they can be used with the standard DebertaV2 huggingface architecture on the normal
.forward() after training without any issues.
"""

# imports
import argparse
from math import log2
from random import randint, random
from collections import Counter
from pathlib import Path
from typing import Optional

# packages
import httpx
import torch
from transformers import (
    DebertaV2Config,
    PreTrainedTokenizerFast,
)

# project
from kl3m_embeddings.embeddings.deberta.matroyshka_deberta import (
    MatroyshkaDebertaV2ForMaskedLM,
)
from kl3m_embeddings.embeddings.trainer import KL3MTorchTrainer

# constants
DEFAULT_TOKENIZER = "alea-institute/kl3m-003-64k"
DEFAULT_ENDPOINT = "http://localhost:8000"


class KL3MDebertaTrainer(KL3MTorchTrainer):
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
            print(f"Error closing endpoint client during cleanup: {str(e)}")

    def setup_model(
        self,
        tokenizer: PreTrainedTokenizerFast,
        precision: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        Get the model.

        Args:
            tokenizer (PreTrainedTokenizerFast): The tokenizer.
            precision (torch.dtype): The precision.
        """
        # try to load from the checkpoint path
        if self.checkpoint_path.exists():
            try:
                self.model = MatroyshkaDebertaV2ForMaskedLM.from_pretrained(
                    self.checkpoint_path,
                )
                print("Loaded model from checkpoint.")
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error loading model from checkpoint: {str(e)}")

        if self.model is None:
            # load from config path
            c2 = DebertaV2Config.from_pretrained(self.config_path)

            # set the vocab size and max position embeddings
            c2.vocab_size = len(self.tokenizer)
            self.model = MatroyshkaDebertaV2ForMaskedLM(c2)

            print("Created new model.")

        # set the precision and device
        self.model.to(device="cuda").to(dtype=precision)

    def get_sample(self, device: Optional[str] = "cuda") -> dict[str, torch.Tensor]:
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
                    mlm_url, json=mlm_post_body, timeout=0.1
                )
                if response.status_code == 200:
                    break

                if response.status_code not in (503,):
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

        # count the datasets and add them to the current entry
        dataset_list = [r["dataset_id"] for r in data]
        dataset_count = Counter(dataset_list)
        self.current_entry["datasets"] = dataset_count

        # update the current log entry
        object_list = [r["identifier"] for r in data]
        self.current_entry["objects"] = object_list

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
            self.current_entry["reduced_dim"] = result["reduced_dim"]

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
    trainer = KL3MDebertaTrainer(
        config_path=args.config_path,
        tokenizer_name=args.tokenizer_name,
        checkpoint_path=args.checkpoint_path,
    )

    # train the model
    trainer.train()
