"""
Extends the generic training wrapper to provide deepspeed model training capabilities.
"""

# imports
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# packages
import deepspeed
import deepspeed.checkpoint
import torch
import tqdm
from transformers.modeling_outputs import MaskedLMOutput

# project
from kl3m_embeddings.embeddings.trainer import (
    DEFAULT_LR,
    DEFAULT_NUM_EVAL_SAMPLES,
    DEFAULT_STEPS_PER_EPOCH,
    DEFAULT_STEPS_PER_EVAL,
    DEFAULT_STEPS_PER_SAVE,
    DEFAULT_TOTAL_STEPS,
    KL3MTorchTrainer,
)


# pylint: disable=too-many-positional-arguments
class KL3MDeepspeedTrainer(KL3MTorchTrainer):
    """
    KL3M Deepspeed Trainer
    """

    INITIALIZE_ON_DEVICE = False

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Optional[Path] = None,
        device: str = "cuda",
        num_workers: int = 2,
    ):
        """
        Initialize the KL3MTrainer.

        Args:
            config_path (Path): The path to the model configuration.
            checkpoint_path (Path): The path to save model checkpoints.
            device (str): The device to use.
            num_workers (int): The number of workers to use in background tasks.
        """
        # load this early
        self.training_config = self.load_training_config(config_path)

        # set config
        if "deepspeed" in self.training_config:
            self.deepspeed_config = self.training_config["deepspeed"]
            if (
                "train_batch_size" not in self.deepspeed_config
                and "train_micro_batch_size_per_gpu" not in self.deepspeed_config
            ):
                self.deepspeed_config["train_micro_batch_size_per_gpu"] = (
                    self.training_config.get("batch_size", 1)
                )
        else:
            self.deepspeed_config = {
                "train_micro_batch_size_per_gpu": self.training_config.get(
                    "batch_size", 1
                ),
                "gradient_accumulation_steps": self.training_config.get(
                    "optimizer", {}
                ).get("gradient_accumulation_steps", 1),
                "optimizer_type": "fused_adamw",
            }

            if self.precision in (torch.bfloat16,):
                self.deepspeed_config["bf16"] = {
                    "enabled": True,
                }
            elif self.precision in (torch.float16,):
                self.deepspeed_config["fp16"] = {
                    "enabled": True,
                }

        # initialize the zero stage as we need it for the optimizer setup case handling
        self.zero_stage = self.deepspeed_config.get("zero_optimization", {}).get(
            "stage", 0
        )

        # call super
        super().__init__(config_path, checkpoint_path, device, num_workers)

        # then override the deepspeed logger
        deepspeed.logger = self.logger

        # set up init args
        ds_args = {
            "config": self.deepspeed_config,
            "model": self.model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "model_parameters": self.model.parameters(),  # type: ignore
        }

        # now initialize the deepspeed engine; skip training data loader
        self.deepspeed_engine, self.deepspeed_optimizer, _, self.deepspeed_scheduler = (
            deepspeed.initialize(
                args={"local_rank": self.local_rank},
                **ds_args,
            )
        )

    def setup_optimizer(
        self,
    ) -> None:
        """
        Get the optimizer, which defaults to AdamW.

        Args:

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        # get deepspeed config
        deepspeed_config = self.training_config.get("deepspeed", {})

        # check if we have set "optimizer_type"
        optimizer_type = deepspeed_config.get("optimizer_type", None)

        # check bad cases to fail on
        # check old notes and new gh issues

        # set the optimizer
        if optimizer_type in ("fused_adamw", "adamw") and self.zero_stage < 2:
            self.optimizer = deepspeed.ops.adam.FusedAdam(
                self.model.parameters(),  # type: ignore
                lr=self.training_config.get("optimizer", {}).get("peak_lr", DEFAULT_LR),
            )
        else:
            # set DeepSeedCPUAdam to be safe
            self.optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
                self.model.parameters(),  # type: ignore
                lr=self.training_config.get("optimizer", {}).get("peak_lr", DEFAULT_LR),
            )

        self.log("Initialized optimizer: %s", self.optimizer)

    def forward(self, **inputs: Any) -> MaskedLMOutput:
        """
        Override the torch forward with the DS engine.

        Args:
            **inputs: The input data

        Returns:
            MaskedLMOutput
        """
        # track timing
        start_time = time.time()

        # log begin
        self.log("Beginning forward pass")

        # get the model outputs
        output = self.deepspeed_engine(**inputs)

        # set loss into current entry
        self.loss_ts.append(output.loss.detach().item())
        self.step_entry["loss"] = self.loss_ts[-1]

        # track timing
        end_time = time.time()

        self.step_entry["forward_time"] = end_time - start_time

        # log end
        self.log(
            "Completed forward pass with loss=%0.2f in %0.2f seconds",
            self.step_entry["loss"],
            self.step_entry["forward_time"],
        )

        return output

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """
        Override the torch backward with the DS engine.

        Args:
            loss (torch.Tensor): The loss.
        """
        # log it
        self.log("Beginning backward pass with loss=%0.2f", loss)

        # track timing
        start_time = time.time()

        # backward pass with kwargs
        self.deepspeed_engine.backward(loss, **kwargs)

        # track timing
        end_time = time.time()

        # log it
        self.log("Completed backward pass in %0.2f seconds", end_time - start_time)

        self.step_entry["backward_time"] = end_time - start_time

    def step(self) -> None:
        """
        Override the torch step with the DS engine.

        Args:

        Returns:
            None
        """
        # track timing
        start_time = time.time()

        # log
        self.log("Beginning optimizer step")

        # step the optimizer
        self.deepspeed_engine.step()

        # track timing
        end_time = time.time()

        # log it
        self.log("Completed optimizer step in %0.2f seconds", end_time - start_time)

        self.step_entry["optimizer_time"] = end_time - start_time

    def save(self) -> None:
        """
        Save the model.

        pipeline.
        """
        # save the model
        if self.zero_stage < 3:
            model_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(
                self.deepspeed_engine.module.state_dict()
            )

            self.log("Saving model to %s", self.checkpoint_path)
            self.deepspeed_engine.module.save_pretrained(
                save_directory=self.checkpoint_path,
                state_dict=model_state_dict,
            )  # type: ignore
        else:
            # saving with 16bit model
            self.log("Saving model to %s", self.checkpoint_path)
            self.deepspeed_engine.module.save_pretrained(
                save_directory=self.checkpoint_path,
                safe_serialization=False,
                max_shard_size="4GB",
            )
            self.deepspeed_engine.save_16bit_model(
                save_dir=self.checkpoint_path,
                save_filename="pytorch_model.bin",
            )
        self.log("Saving tokenizer to %s", self.checkpoint_path)
        self.tokenizer.save_pretrained(self.checkpoint_path)  # type: ignore

    def convert_to_safetensors(self) -> None:
        """
        Load a pytorch_model.bin model and convert to new safetensors format after
        collecting state_dict from stage3 training run.

        Returns:
            None
        """
        # load the model
        torch_model = self.model.from_pretrained(  # type: ignore
            self.checkpoint_path, use_safetensors=False
        )

        # convert to safetensors
        torch_model.save_pretrained(
            save_directory=self.checkpoint_path, use_safetensors=True
        )

        # unlink the old model
        model_path = self.checkpoint_path / "pytorch_model.bin"
        if model_path.exists():
            model_path.unlink()

    def get_grad_norm(self) -> float:
        """
        Safely get the gradient norm under deepspeed, which is
        a bit trickier than plain torch.

        Returns:
            float: The gradient norm.
        """
        try:
            grad_tensors = torch.stack(
                [
                    deepspeed.utils.safe_get_full_grad(param).norm()
                    for param in self.deepspeed_engine.parameters()
                ]
            )
            grad_norm = torch.norm(grad_tensors.detach()).item()
        except Exception:  # pylint: disable=broad-except
            grad_norm = 0.0

        return grad_norm

    # pylint: disable=too-many-statements,too-many-branches
    def train(self, steps: Optional[int] = None) -> bool:
        """
        Train the model.

        Args:
            steps (int): The number of steps to train for.

        Returns:
            bool: Whether the training was successful.
        """
        # start populating the queue
        for _ in range(self.sample_queue.maxsize):
            self.sample_thread_pool.submit(self.populate_queue)

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

        # populate the eval sample
        print("Populating eval samples...")
        self.load_eval_data(num_eval_samples)
        print(f"Populated {len(self.eval_samples)} eval samples.")

        # NB: we can't reliably use torch.compile with deepspeed
        try:
            # compile the whole model here
            self.log("Beginning model compilation...")
            self.model = torch.compile(self.model, fullgraph=True).to(
                device=self.device, dtype=self.precision
            )
            self.log("Model compiled and moved to device")
        except Exception as e:  # pylint: disable=broad-except
            self.log("Error compiling model: %s", str(e), level="error")

        # start training loop
        train_start_time = time.time()
        train_status = False
        try:
            if self.local_rank == 0:
                prog_bar = tqdm.tqdm(
                    initial=step,
                    desc="Training",
                    total=steps or total_steps,
                )
            else:
                prog_bar = None
            while step < (steps or total_steps):
                # log
                self.log("Beginning step %d", step)
                self.scheduler.current_step = step  # type: ignore

                # populate more samples in the thread pool
                for _ in range(self.sample_queue.maxsize):
                    self.sample_thread_pool.submit(self.populate_queue)

                # update the entry
                step_start_time = time.time()
                self.step_entry["step"] = step
                self.step_entry["epoch"] = epoch
                self.step_entry["lr"] = self.get_lr()

                # get the sample
                sample_start_time = time.time()
                while True:
                    try:
                        sample, metadata = self.get_next_sample()
                        self.step_entry.update(metadata)
                        break
                    except Exception as e:  # pylint: disable=broad-except
                        self.log("Error getting sample: %s", str(e), level="error")
                        self.log(traceback.format_exc(), level="error")
                        time.sleep(1)
                sample_end_time = time.time()
                self.step_entry["sample_time"] = sample_end_time - sample_start_time

                # forward pass
                outputs = self.forward(**sample)

                # backward pass
                self.backward(outputs.loss)

                # add the raw norm to the time series
                grad_norm = self.get_grad_norm()

                if step % gradient_accumulation_steps == 0 and step > 0:
                    self.loss_norm_ts.append(grad_norm)

                # clip (if configured)
                self.clip_grad(grad_norm)

                # step optimizer
                self.step()

                # log training metrics
                if step % steps_per_save == 0 and step > 0:
                    self.save()

                # check if we are doing eval
                if step % steps_per_eval == 0 and step > 0:
                    eval_results = self.eval()
                    self.eval_ts.append(eval_results["mean"])

                # get final time
                step_end_time = time.time()
                self.step_entry["step_time"] = step_end_time - step_start_time

                # get the time from end of step to start of training
                self.step_entry["total_time"] = step_end_time - train_start_time

                # calculate the rate from total number of tokens
                total_tokens += self.step_entry.get("num_tokens", 0)
                token_rate = total_tokens / max(self.step_entry["total_time"], 1.0)
                self.step_entry["token_rate"] = token_rate

                # get trailing loss
                last_eval = self.eval_ts[-1] if len(self.eval_ts) > 0 else 0.0
                if prog_bar:
                    prog_bar.set_postfix(
                        {
                            "loss": f"{self.step_entry.get('loss', 99.9):0.2f}",
                            "loss_100": f"{self.get_trailing_loss(100):0.3f}",
                            "loss_1000": f"{self.get_trailing_loss(1000):0.3f}",
                            "last_eval": f"{last_eval:0.2f}",
                            "grad_norm": f"{grad_norm:0.2f}",
                            "lr": f"{self.get_lr():1.1e}",
                            "step_time": f"{self.step_entry['step_time']:0.2f}",
                            "token_rate": f"{token_rate:0.2f}",
                        }
                    )
                    prog_bar.update(1)

                # log the entry
                self.log_step()

                # inc the epoch
                if step % steps_per_epoch == 0:
                    epoch += 1

                step += 1

                # log it
                self.log("Completed step %d", step)

            train_status = True
        except KeyboardInterrupt:
            print("Interrupted training")
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error during training: {e}")
            traceback.print_exc()
            self.log("Error during training: %s", str(e), level="error")
            self.log(traceback.format_exc(), level="error")
        finally:
            # final save
            self.save()

            # check deepspeed stage
            if self.zero_stage == 3:
                self.convert_to_safetensors()

            # force thread pool shutdown after saving
            try:
                self.sample_thread_pool.shutdown(wait=False)
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error shutting down sample thread pool: {e}")

        return train_status
