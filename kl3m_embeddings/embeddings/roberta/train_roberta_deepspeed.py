"""
Train a kl3m roberta model with given configuration using deepspeed, which supports
both distributed training and various optimization and parallelization strategies.

Note that these models are trained with Matroyshka-style dimensionality reduction, but
they can be used with the standard Roberta huggingface architecture on the normal
.forward() after training without any issues.
"""

# imports
import argparse
from math import log2
from pathlib import Path
from random import randint, random
from typing import Optional

# packages
import httpx
import torch
from transformers import PreTrainedTokenizerFast, RobertaConfig

# project
from kl3m_embeddings.embeddings.deepspeed_trainer import KL3MDeepspeedTrainer
from kl3m_embeddings.embeddings.roberta.matroyshka_roberta import (
    MatroyshkaRobertaForMaskedLM,
)
from kl3m_embeddings.embeddings.samples.embedding import get_embedding_sample
from kl3m_embeddings.utils.models import get_model_size_str

# constants
DEFAULT_TOKENIZER = "alea-institute/kl3m-003-64k"
DEFAULT_ENDPOINT = "http://localhost:8000"


class KL3MDeepspeedRobertaTrainer(KL3MDeepspeedTrainer):
    """
    Trainer for kl3m roberta model.
    """

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path = Path("checkpoint/"),
    ):
        # call the super
        super().__init__(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )

        # get task probabilities
        self.task_probabilities = self.training_config.get(
            "tasks", {"mlm": 0.5, "nsp": 0.5}
        )

        # get the matroyshka sampling probability
        self.matroyshka_probability = self.training_config.get(
            "matroyshka_probability", 0.5
        )
        self.matroyshka_min_log2 = self.training_config.get("matroyshka_min_log2", 3)

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
        self.log("Using precision: %s", precision)

        # try to load from the checkpoint path
        if self.checkpoint_path.exists():
            try:
                self.model = MatroyshkaRobertaForMaskedLM.from_pretrained(
                    self.checkpoint_path,
                    torch_dtype=precision,
                )
                self.model.train()
                self.log("Loaded model from checkpoint.")
            except Exception as e:  # pylint: disable=broad-except
                self.log(
                    "Error loading model from checkpoint: %s", str(e), level="error"
                )

        if self.model is None:
            # load from config path
            model_config = RobertaConfig.from_pretrained(
                self.config_path,
                torch_dtype=precision,
            )

            # set the vocab size and max position embeddings
            model_config.vocab_size = len(self.tokenizer)
            self.model = MatroyshkaRobertaForMaskedLM(model_config)
            self.model.train()

            self.log("Created model from config.")

        # set the model to the device and precision
        self.model.train()
        if self.INITIALIZE_ON_DEVICE:
            self.model.to(dtype=precision).to(self.device)
        else:
            self.model.to(dtype=precision)

    def get_sample(self, device: Optional[str] = "cpu") -> dict[str, torch.Tensor]:
        """
        Get a sample for training.

        Returns:
            dict[str, torch.Tensor]: The sample.
        """
        # get a task based on probability dist
        result, sample_metadata = get_embedding_sample(
            task_probabilities=self.task_probabilities,
            batch_size=self.training_config.get("batch_size", 1),
            endpoint_url=self.endpoint_url,
            endpoint_client=self.endpoint_client,
            logger=self.logger,
            device=device or self.device,
        )

        # get matroyshka samples
        if random() < self.matroyshka_probability:
            max_log2 = int(log2(self.model.config.hidden_size)) - 1  # type: ignore
            result["reduced_dim"] = 2 ** randint(self.matroyshka_min_log2, max_log2)
            self.step_entry["reduced_dim"] = result["reduced_dim"]

        # update step entry
        self.step_entry.update(sample_metadata)

        # handle matroyshka sampling
        if random() < self.matroyshka_probability:
            max_log2 = int(log2(self.model.config.hidden_size)) - 1  # type: ignore
            result["reduced_dim"] = 2 ** randint(self.matroyshka_min_log2, max_log2)
            self.step_entry["reduced_dim"] = result["reduced_dim"]

        return result

    def load_eval_data(self, num_samples: int) -> None:
        """
        Get evaluation data for the model.

        Args:
            num_samples (int): The number of samples to get.
        """
        # get the samples
        while len(self.eval_samples) < num_samples:
            result, _ = get_embedding_sample(
                task_probabilities=self.task_probabilities,
                batch_size=1,
                endpoint_url=self.endpoint_url,
                endpoint_client=self.endpoint_client,
                logger=self.logger,
                device="cpu",
            )

            if result:
                self.eval_samples.append(result)


if __name__ == "__main__":
    # set up args
    parser = argparse.ArgumentParser(
        description="Train a roberta model with given configuration."
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
    # add deepspeed args now for rank
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed from deepspeed",
    )
    args = parser.parse_args()

    # create the trainer
    trainer = KL3MDeepspeedRobertaTrainer(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )

    # print key tokenizer and model info
    print(f"Tokenizer Size: {len(trainer.tokenizer)}")
    print(f"Model Size: {get_model_size_str(trainer.model)}")
    print(f"Training Precision: {trainer.precision}")

    # train the model
    trainer.train()
