"""
InfoNCE contrastive trainer for kl3m roberta model.
"""

# imports
import argparse
import base64
import traceback
import time
import zlib
from pathlib import Path
from random import randint, choice, shuffle
from typing import Optional, Any

# packages
import httpx
import torch
import torch.nn.functional as F
import tqdm
from transformers import RobertaConfig, PreTrainedTokenizerFast

# project
from kl3m_embeddings.models.roberta.matroyshka_roberta import (
    MatroyshkaRobertaForMaskedLM,
)
from kl3m_embeddings.models.samples import get_embedding_sample, get_random_source
from kl3m_embeddings.models.trainer import (
    KL3MTorchTrainer,
    DEFAULT_STEPS_PER_EPOCH,
    DEFAULT_STEPS_PER_SAVE,
    DEFAULT_STEPS_PER_EVAL,
    DEFAULT_NUM_EVAL_SAMPLES,
    DEFAULT_TOTAL_STEPS,
)
from kl3m_embeddings.utils.models import get_model_size_str

# constants
DEFAULT_TOKENIZER = "alea-institute/kl3m-004-128k-uncased"
DEFAULT_ENDPOINT = "http://localhost:8000"


class KL3MRobertaContrastiveTrainer(KL3MTorchTrainer):
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

    def get_sample(
        self, device: Optional[str] = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """
        Get a sample for training.

        Args:
            device (str): The device to use.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any]]: The sample and metadata.
        """
        # get the device
        device = device or self.device

        return {}, {}

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

    def get_token_chunks(
        self,
        tokens: list[int],
    ) -> list[list[int]]:
        """
        Get the token chunks for the given tokens.

        Args:
            tokens (list[int]): The tokens.

        Returns:
            list[list[int]]: The token chunks.
        """
        max_size = self.model.config.max_position_embeddings - 2
        chunks = []
        for start_index in range(0, len(tokens), max_size):
            chunks.append(tokens[start_index : start_index + max_size])
        return chunks

    def prepare_sample(self, tokens: list[int]) -> dict[str, torch.Tensor]:
        """
        Prepare a sample for training with just input_ids and attention_mask.

        Args:
            tokens (list[int]): The tokens.

        Returns:
            dict[str, torch.Tensor]: The sample.
        """
        # get lengths
        sample_length = len(tokens)
        pad_length = self.model.config.max_position_embeddings - sample_length - 2

        input_ids = [
            [self.tokenizer.cls_token_id]
            + tokens
            + [self.tokenizer.sep_token_id]
            + [self.tokenizer.pad_token_id] * pad_length
        ]

        attention_mask = [[1] + [1] * sample_length + [1] + [0] * pad_length]

        return {
            "input_ids": torch.tensor(input_ids, device=self.device),
            "attention_mask": torch.tensor(attention_mask, device=self.device),
        }

    def get_sample_batch(
        self,
        num_per_pair: int = 3,
        min_datasets: int = 3,
        min_samples_per_dataset: int = 3,
    ) -> list[
        tuple[
            dict[str, torch.Tensor],
            dict[str, torch.Tensor],
            dict[str, torch.Tensor],
            dict[str, torch.Tensor],
        ]
    ]:
        """
        Get a batch of samples for contrastive loss with same, same, similar, different.

        Returns:
            list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]]: The samples in quads.
        """
        sample_data: list[
            tuple[
                dict[str, torch.Tensor],
                dict[str, torch.Tensor],
                dict[str, torch.Tensor],
                dict[str, torch.Tensor],
            ]
        ] = []

        samples_by_dataset: dict[str, list[dict[str, Any]]] = {}
        while True:
            # get raw result
            result = get_random_source(
                endpoint_url=trainer.endpoint_url,
                endpoint_client=trainer.endpoint_client,
                device="cpu",
            )

            # decode content
            result["content"] = zlib.decompress(
                base64.b64decode(result["content"])
            ).decode()
            result["tokens"] = self.tokenizer(
                result["content"],
                truncation=False,
                add_special_tokens=False,
            ).input_ids

            dataset_id = result["dataset"]
            if dataset_id not in samples_by_dataset:
                samples_by_dataset[dataset_id] = []
            samples_by_dataset[dataset_id].append(result)

            # check stopping condition
            if len(samples_by_dataset) >= min_datasets:
                if all(
                    len(samples) >= min_samples_per_dataset
                    for samples in samples_by_dataset.values()
                ):
                    break

        # get all combinations of same and different dataset samples
        dataset_list = list(samples_by_dataset.keys())
        for i in range(len(dataset_list)):
            dataset_i = samples_by_dataset[dataset_list[i]]
            for j in range(i):
                dataset_j = samples_by_dataset[dataset_list[j]]

                # get two random chunks from i, one from j
                for _ in range(num_per_pair):
                    # get two random documents from dataset i
                    document_i1 = choice(dataset_i)
                    document_i2 = choice(dataset_i)
                    document_j = choice(dataset_j)
                    # print(document_i1["identifier"], document_i2["identifier"], document_j["identifier"])

                    content_i1 = document_i1["tokens"]
                    content_i2 = document_i2["tokens"]
                    content_j = document_j["tokens"]

                    chunks_i1 = self.get_token_chunks(content_i1)
                    chunks_i2 = self.get_token_chunks(content_i2)
                    chunks_j = self.get_token_chunks(content_j)

                    chunk_i1a = chunks_i1[randint(0, len(chunks_i1) - 1)]
                    chunk_i1b = chunks_i1[randint(0, len(chunks_i1) - 1)]
                    chunk_i2 = chunks_i2[randint(0, len(chunks_i2) - 1)]
                    chunk_j = chunks_j[randint(0, len(chunks_j) - 1)]

                    # append
                    sample_data.append(
                        (
                            self.prepare_sample(chunk_i1a),
                            self.prepare_sample(chunk_i1b),
                            self.prepare_sample(chunk_i2),
                            self.prepare_sample(chunk_j),
                        )
                    )

        # shuffle the order
        shuffle(sample_data)

        return sample_data

    # pylint: disable=too-many-statements,too-many-branches,too-many-positional-arguments
    def train(self, steps: Optional[int] = None) -> bool:
        """
        Train the model.

        Args:
            steps (int): The number of steps to train for.

        Returns:
            bool: Whether the training was successful.
        """
        # get key training params
        steps_per_epoch = self.training_config.get(
            "steps_per_epoch", DEFAULT_STEPS_PER_EPOCH
        )
        steps_per_save = self.training_config.get(
            "steps_per_save", DEFAULT_STEPS_PER_SAVE
        )
        steps_per_eval = self.training_config.get(
            "steps_per_eval", DEFAULT_STEPS_PER_EVAL
        )
        num_eval_samples = self.training_config.get(
            "num_eval_samples", DEFAULT_NUM_EVAL_SAMPLES
        )
        optimizer_config = self.training_config.get("optimizer", {})
        gradient_accumulation_steps = optimizer_config.get(
            "gradient_accumulation_steps", 1
        )
        total_steps = optimizer_config.get("total_steps", DEFAULT_TOTAL_STEPS)

        # get state
        state = self.load_state(self.checkpoint_path)

        # tracking vars
        epoch = state["epoch"]
        step = state["step"]
        total_tokens = 0
        if hasattr(self.scheduler, "current_step"):
            self.scheduler.current_step = step  # type: ignore

        # start training loop
        train_start_time = time.time()
        train_status = False

        # compile the whole model here
        if self.training_config.get("compile", False):
            self.log("Beginning model compilation...")
            self.model = torch.compile(self.model, fullgraph=True, dynamic=True).to(
                device=self.device, dtype=self.precision
            )
            self.log("Model compiled and moved to device")

        # wait to put the model into the right device until now
        with torch.amp.autocast(device_type=self.device, dtype=self.precision):
            try:
                prog_bar = tqdm.tqdm(
                    initial=step,
                    desc="Training",
                    total=steps or total_steps,
                )
                while step < (steps or total_steps):
                    # log
                    self.log("Beginning step %d", step)
                    self.scheduler.current_step = step  # type: ignore

                    # update the entry
                    step_start_time = time.time()
                    self.step_entry["step"] = step
                    self.step_entry["epoch"] = epoch
                    self.step_entry["lr"] = self.get_lr()

                    # get the sample
                    sample_start_time = time.time()
                    while True:
                        try:
                            sample_batch = self.get_sample_batch()
                            break
                        except Exception as e:  # pylint: disable=broad-except
                            self.log("Error getting sample: %s", str(e), level="error")
                            self.log(traceback.format_exc(), level="error")
                            time.sleep(1)
                    sample_end_time = time.time()
                    self.step_entry["sample_time"] = sample_end_time - sample_start_time

                    # we are now going to go through all samples, which are quad of same,same,similar,different.
                    # for each quad, we are going to:
                    # 1. forward pass all three samples to get the last hidden state tensors
                    # 2. dot the same and different tensors to get the cosine similarity
                    # 3. calculate a loss based on same, similar, diff margins
                    # and close to 0 for the same-different tensors.
                    # 4. aggregate the loss across all samples
                    # 5. backward pass the loss
                    # 6. step the optimizer
                    # 7. log the loss

                    # forward pass
                    batch_losses = []
                    sim_same_values = []
                    sim_similar_values = []
                    sim_diff_values = []

                    forward_time = 0.0
                    for sample_triple in sample_batch:
                        # get the forwards with hidden states
                        forward_start_time = time.time()
                        embedding_i1a = self.model(
                            **sample_triple[0], output_hidden_states=True
                        )
                        embedding_i1b = self.model(
                            **sample_triple[1], output_hidden_states=True
                        )
                        embedding_i2 = self.model(
                            **sample_triple[2], output_hidden_states=True
                        )
                        embedding_j = self.model(
                            **sample_triple[3], output_hidden_states=True
                        )
                        forward_end_time = time.time()
                        forward_time += forward_end_time - forward_start_time

                        # each final hidden state is (1, 512, 1024)
                        # we want to apply the attention mask to remove any pad tokens from the second dimension
                        hidden_i1a = (
                            embedding_i1a["hidden_states"][-1][0, :, :]
                            * sample_triple[0]["attention_mask"][0, :, None]
                        ).mean(dim=0)

                        hidden_i1b = (
                            embedding_i1b["hidden_states"][-1][0, :, :]
                            * sample_triple[1]["attention_mask"][0, :, None]
                        ).mean(dim=0)

                        hidden_i2 = (
                            embedding_i2["hidden_states"][-1][0, :, :]
                            * sample_triple[2]["attention_mask"][0, :, None]
                        ).mean(dim=0)

                        hidden_j = (
                            embedding_j["hidden_states"][-1][0, :, :]
                            * sample_triple[3]["attention_mask"][0, :, None]
                        ).mean(dim=0)

                        # Calculate cosine similarities
                        sim_i1a_i1b = F.cosine_similarity(hidden_i1a, hidden_i1b, dim=0)
                        sim_i1a_i2 = F.cosine_similarity(hidden_i1a, hidden_i2, dim=0)
                        sim_i1b_i2 = F.cosine_similarity(hidden_i1b, hidden_i2, dim=0)
                        sim_i1a_j = F.cosine_similarity(hidden_i1a, hidden_j, dim=0)
                        sim_i1b_j = F.cosine_similarity(hidden_i1b, hidden_j, dim=0)
                        sim_i2_j = F.cosine_similarity(hidden_i2, hidden_j, dim=0)

                        sim_same_values.append(sim_i1a_i1b.detach().item())
                        sim_similar_values.extend(
                            [
                                sim_i1a_i2.detach().item(),
                                sim_i1b_i2.detach().item(),
                            ]
                        )
                        sim_diff_values.extend(
                            [
                                sim_i1a_j.detach().item(),
                                sim_i1b_j.detach().item(),
                                sim_i2_j.detach().item(),
                            ]
                        )

                        margin_pos = 1.0
                        margin_mid = 0.0
                        margin_neg = -1.0
                        loss = torch.logsumexp(
                            torch.stack(
                                [
                                    margin_pos - sim_i1a_i1b,
                                    0.5 * (margin_mid - sim_i1a_i2),
                                    0.5 * (margin_mid - sim_i1b_i2),
                                    0.33 * (sim_i1a_j - margin_neg),
                                    0.33 * (sim_i1b_j - margin_neg),
                                    0.33 * (sim_i2_j - margin_neg),
                                ]
                            ),
                            dim=0,
                        )

                        # backward pass
                        self.backward(loss)

                        # set loss into current entry
                        batch_losses.append(loss.detach().item())

                    self.step_entry["forward_time"] = forward_time

                    # track loss
                    self.loss_ts.append(sum(batch_losses) / len(batch_losses))
                    self.step_entry["loss"] = self.loss_ts[-1]

                    # add the raw norm to the time series
                    grad_norm = self.get_grad_norm()
                    if (step + 1) % gradient_accumulation_steps == 0 and step > 0:
                        self.loss_norm_ts.append(grad_norm)

                    # clip (if configured)
                    if (step + 1) % gradient_accumulation_steps == 0 and step > 0:
                        self.clip_grad(grad_norm)
                        self.step()

                    # log training metrics
                    if step % steps_per_save == 0 and step > 0:
                        self.save()

                    # get final time
                    step_end_time = time.time()
                    self.step_entry["step_time"] = step_end_time - step_start_time

                    # get the time from end of step to start of training
                    self.step_entry["total_time"] = step_end_time - train_start_time

                    # calculate the rate from total number of tokens
                    total_tokens += self.step_entry.get("num_tokens", 0)
                    token_rate = total_tokens / max(self.step_entry["total_time"], 1.0)
                    self.step_entry["token_rate"] = token_rate

                    # get mean same, similar, and diff values
                    mean_sim_same = sum(sim_same_values) / len(sim_same_values)
                    mean_sim_similar = sum(sim_similar_values) / len(sim_similar_values)
                    mean_sim_diff = sum(sim_diff_values) / len(sim_diff_values)

                    # get trailing loss
                    last_eval = self.eval_ts[-1] if len(self.eval_ts) > 0 else 0.0
                    prog_bar.set_postfix(
                        {
                            "loss": f"{self.step_entry.get('loss', 99.9):0.2f}",
                            "loss_100": f"{self.get_trailing_loss(100):0.3f}",
                            "loss_1000": f"{self.get_trailing_loss(1000):0.3f}",
                            "grad_norm": f"{grad_norm:0.2f}",
                            "lr": f"{self.get_lr():1.1e}",
                            "last_eval": f"{last_eval:0.2f}",
                            "step_time": f"{self.step_entry['step_time']:0.2f}",
                            "token_rate": f"{token_rate:0.2f}",
                            "mean_sim_same": f"{mean_sim_same:0.2f}",
                            "mean_sim_similar": f"{mean_sim_similar:0.2f}",
                            "mean_sim_diff": f"{mean_sim_diff:0.2f}",
                        }
                    )

                    # log the entry
                    self.log_step()

                    # inc the epoch
                    if step % steps_per_epoch == 0:
                        epoch += 1

                    step += 1
                    prog_bar.update(1)

                    # log it
                    self.log("Completed step %d", step)

                train_status = True
            except KeyboardInterrupt:
                print("Interrupted training")
            except Exception as e:  # pylint: disable=broad-except
                print(
                    f"Error during training on {self.local_rank}/{self.global_rank}: {e}"
                )
                traceback.print_exc()
                self.log("Error during training: %s", str(e), level="error")
                self.log(traceback.format_exc(), level="error")
            finally:
                # final save
                self.save()

                # force thread pool shutdown after saving
                try:
                    self.sample_thread_pool.shutdown(wait=False, cancel_futures=True)
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error shutting down sample thread pool: {e}")

        return train_status


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
    args = parser.parse_args()

    # create the trainer
    trainer = KL3MRobertaContrastiveTrainer(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )

    # print key tokenizer and model info
    print(f"Tokenizer Size: {len(trainer.tokenizer)}")
    print(f"Model Size: {get_model_size_str(trainer.model)}")
    print(f"Training Precision: {trainer.precision}")

    # train the model
    trainer.train()
