"""
Train a kl3m roberta model with given configuration on a single node, single GPU setup
with pure torch.  transformers is only used to load the model architecture and tokenizer.

Note that these models are trained with Matroyshka-style dimensionality reduction, but
they can be used with the standard Roberta huggingface architecture on the normal
.forward() after training without any issues.
"""

# imports
import argparse
import datetime
import os
import time
from math import log2
from pathlib import Path
from random import randint, random
from typing import Optional, Any

# packages
import httpx
import torch
from transformers import RobertaConfig, PreTrainedTokenizerFast


# project
from kl3m_embeddings.models.trainer import KL3MTorchTrainer
from kl3m_embeddings.optimizers.muon_modified import Muon
from kl3m_embeddings.models.roberta.matroyshka_roberta import (
    MatroyshkaRobertaForMaskedLM,
)
from kl3m_embeddings.models.samples import get_embedding_sample
from torch.nn.parallel import DistributedDataParallel

from kl3m_embeddings.utils.logger import LOGGER
from kl3m_embeddings.utils.models import get_model_size_str

# constants
DEFAULT_TOKENIZER = "alea-institute/kl3m-004-128k-uncased"
DEFAULT_ENDPOINT = "http://localhost:8000"

# get the environment variables at the module level
MASTER_ADDR = os.environ.get("MASTER_ADDR", "localhost")
MASTER_PORT = int(os.environ.get("MASTER_PORT", 29500))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


class KL3MDebertaTrainer(KL3MTorchTrainer):
    """
    Trainer for kl3m roberta model.
    """

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path = Path("checkpoint/"),
    ):
        # get all vars up front here
        self.master_addr = os.getenv("MASTER_ADDR", "localhost")
        self.master_port = int(os.getenv("MASTER_PORT", "29500"))
        self.global_rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        # set device
        print("Setting device", self.local_rank)
        torch.cuda.set_device(self.local_rank)

        # init_process_group
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            world_size=self.world_size,
            rank=self.global_rank,
            timeout=datetime.timedelta(seconds=3600),
        )

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
        """
        precision = precision or self.precision
        self.log("Using precision: %s", precision)

        if self.checkpoint_path.exists():
            try:
                self.model = MatroyshkaRobertaForMaskedLM.from_pretrained(
                    self.checkpoint_path,
                    torch_dtype=precision,
                )
                self.log("Loaded model from checkpoint.")
            except Exception as e:
                self.log(
                    "Error loading model from checkpoint: %s", str(e), level="error"
                )

        if self.model is None:
            model_config = RobertaConfig.from_pretrained(
                self.config_path,
                torch_dtype=precision,
            )
            model_config.vocab_size = len(self.tokenizer)
            self.model = MatroyshkaRobertaForMaskedLM(model_config)

        # Move model to device first
        self.model.to(device=f"cuda:{self.local_rank}", dtype=precision)

        # Then wrap in DDP
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )

        # compile here if requested

        if self.training_config.get("compile", False):
            self.model = torch.compile(self.model)

        self.model.train()

    def setup_optimizer(self) -> None:
        """Get the optimizer."""
        # Set up optimizer after DDP wrapping
        optimizer_config = self.training_config.get("optimizer", {})

        # sort params by optimizer
        muon_params = []
        adamw_params = []
        for name, param in self.model.module.named_parameters():
            if "embed" in name or "head" in name or param.ndim < 2:
                adamw_params.append(param)
            else:
                muon_params.append(param)

        self.optimizer = Muon(
            muon_params,
            lr=optimizer_config.get("muon_lr", 0.001),
            momentum=optimizer_config.get("momentum", 0.9),
            adamw_params=adamw_params,
            adamw_lr=optimizer_config.get("peak_lr", 3e-4),
            adamw_betas=(
                optimizer_config.get("adamw_beta1", 0.9),
                optimizer_config.get("adamw_beta2", 0.95),
            ),
            adamw_wd=optimizer_config.get("adamw_wd", 0.01),
        )

    def step(self) -> None:
        """
        Optimizer step with gradient accumulation handling
        """
        optimizer_config = self.training_config.get("optimizer", {})
        gradient_accumulation_steps = optimizer_config.get(
            "gradient_accumulation_steps", 1
        )

        # Only step optimizer and scheduler after accumulating enough gradients
        if (self.step_entry.get("step", 0) + 1) % gradient_accumulation_steps == 0:
            start_time = time.time()

            # only do this on rank 0
            self.log("Optimizer step %d", self.step_entry.get("step", 0))

            # reduce the gradient
            for param_name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    reduce_start_time = time.time()
                    torch.distributed.all_reduce(
                        param.grad.data, op=torch.distributed.ReduceOp.AVG
                    )
                    reduce_end_time = time.time()
                    LOGGER.info(
                        "Reduced time on %d: %s @ %f",
                        self.local_rank,
                        param_name,
                        reduce_end_time - reduce_start_time,
                    )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            end_time = time.time()
            self.step_entry["optimizer_time"] = end_time - start_time
        else:
            self.step_entry["optimizer_time"] = 0.0

    def get_sample(
        self, device: Optional[str] = "cpu"
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
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
            max_log2 = int(log2(self.model.module.config.hidden_size)) - 1  # type: ignore
            result["reduced_dim"] = 2 ** randint(self.matroyshka_min_log2, max_log2)
            sample_metadata["reduced_dim"] = result["reduced_dim"]

        # handle matroyshka sampling
        if random() < self.matroyshka_probability:
            max_log2 = int(log2(self.model.module.config.hidden_size)) - 1  # type: ignore
            result["reduced_dim"] = 2 ** randint(self.matroyshka_min_log2, max_log2)
            sample_metadata["reduced_dim"] = result["reduced_dim"]

        return result, sample_metadata

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

    def save(self) -> None:
        """
        Save the model.
        """
        # save the model
        self.log("Saving model to %s", self.checkpoint_path)

        # barrier here
        torch.distributed.barrier()

        # create a new model from the current config,
        # then load state_dict() from ddp model
        save_model = MatroyshkaRobertaForMaskedLM(self.model.module.config)
        save_model.load_state_dict(
            {k.replace("module.", ""): v for k, v in self.model.state_dict().items()}
        )

        # set precision
        save_model.to(dtype=self.precision)

        # save the model
        save_model.save_pretrained(self.checkpoint_path, safe_serialization=False)  # type: ignore

        self.log("Saving tokenizer to %s", self.checkpoint_path)
        self.tokenizer.save_pretrained(self.checkpoint_path)  # type: ignore


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
    parser.add_argument("--rank", type=int, default=RANK, help="Rank of the process")
    parser.add_argument(
        "--local_rank", type=int, default=LOCAL_RANK, help="Local rank of the process"
    )
    parser.add_argument(
        "--world_size", type=int, default=WORLD_SIZE, help="World size of the process"
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default=MASTER_ADDR,
        help="Master address of the process",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=MASTER_PORT,
        help="Master port of the process",
    )
    args = parser.parse_args()

    RANK = args.rank
    LOCAL_RANK = args.local_rank
    WORLD_SIZE = args.world_size
    MASTER_ADDR = args.master_addr
    MASTER_PORT = args.master_port

    # create the trainer
    trainer = KL3MDebertaTrainer(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )

    # print key tokenizer and model info
    print(f"Tokenizer Size: {len(trainer.tokenizer)}")
    print(f"Model Size: {get_model_size_str(trainer.model)}")
    print(f"Training Precision: {trainer.precision}")

    # train the model
    trainer.train()
