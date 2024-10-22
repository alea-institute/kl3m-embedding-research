"""
Generic training wrapper designed to standardize the setup and logging of a model training loop
for embedding models.

This can be used to train models on single GPU, multi-GPU, and multi-node setups with
pure torch, accelerate, deepspeed, etc.
"""

# imports
import abc
import atexit
import concurrent.futures
import datetime
import gzip
import json
import logging
import os
import queue
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# packages
import torch.nn
import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import MaskedLMOutput

# project
from kl3m_embeddings.embeddings.stats.svds import get_layer_svd_stats
from kl3m_embeddings.schedulers.exponential_warmup_cooldown import (
    ExponentialWarmupCooldownScheduler,
)
from kl3m_embeddings.schedulers.linear_warmup_cooldown import (
    LinearWarmupCooldownScheduler,
)
from kl3m_embeddings.utils.logger import get_logger

# constants
DEFAULT_LR = 1e-4
DEFAULT_WARMUP_STEPS = 10000
DEFAULT_TOTAL_STEPS = 100000
DEFAULT_STEPS_PER_EPOCH = 1000000
DEFAULT_STEPS_PER_SAVE = 1000
DEFAULT_STEPS_PER_EVAL = 1000
DEFAULT_NUM_EVAL_SAMPLES = 10000
DEFAULT_ENDPOINT = "http://localhost:8000"
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_MAX_GRAD_NORM_PCT = 90
DEFAULT_SAMPLE_SLEEP = 10.0


# pylint: disable=too-many-instance-attributes,too-many-positional-arguments,too-many-public-methods
class KL3MTorchTrainer(abc.ABC):
    """
    Base KL3M torch-only trainer class that wraps:
    1. loading model configuration
    2. loading training configuration
    3. instantiating a tokenizer
    4. instantiating a model
    5. setting up optimizer and scheduler
    6. setting up training loop
    7. logging training metrics and losses
    8. saving model checkpoints

    No deepspeed, accelerate, or transformers.Trainer here.
    """

    INITIALIZE_ON_DEVICE = True

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
        # create an instance logger with the class name
        self.logger = get_logger(self.__class__.__name__)
        self.global_rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        # get the config and checkpoint paths
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path or config_path

        # set the device
        self.device = device

        # if cuda, set float32 matmul precision
        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")

        # load the training config
        self.training_config = self.load_training_config(config_path)
        self.tokenizer_name = self.training_config.get(
            "tokenizer", "alea-institute/kl3m-003-64k"
        )

        # set precision
        self.precision = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.training_config.get("precision", "bfloat16"), torch.bfloat16)

        # get the tokenizer and model
        self.tokenizer = self.get_tokenizer(self.tokenizer_name)
        self.model: Optional[torch.nn.Module] = None
        self.setup_model(
            tokenizer=self.tokenizer,
        )

        # get the optimizer and scheduler
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

        # set up optional sample queue for models that can lookahead
        # NB: be careful if you increase this ratio, as extra samples are stored on the target device
        # by default and you may thus OOM VRAM unexpectedly.
        self.sample_queue: queue.Queue[dict] = queue.Queue(maxsize=num_workers)

        # set up a default thread pool for sampling
        self.sample_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="kl3m_trainer",
        )

        # start populating the queue before the optimizer and scheduler
        for _ in range(self.sample_queue.maxsize):
            self.sample_thread_pool.submit(self.populate_queue)

        # load them
        self.setup_optimizer()
        self.setup_scheduler()

        # open the log path and setup log step structure
        self.log_path = self.checkpoint_path / "log.jsonl"
        self.log_path.touch()
        self.log_file = self.log_path.open("at+", encoding="utf-8")
        self.eval_path = self.checkpoint_path / "eval.jsonl"
        self.eval_path.touch()
        self.eval_file = self.eval_path.open("at+", encoding="utf-8")
        self.object_log_path = self.checkpoint_path / "objects.jsonl.gz"
        self.object_log_path.touch()
        self.object_log_file = gzip.open(self.object_log_path, "at+", encoding="utf-8")

        # store eval samples on cpu permanently
        self.eval_samples: list[dict] = []

        # setup log entry structures
        self.step_entry: dict[str, Any] = {}
        self.step_logs: list[dict] = []
        self.loss_ts: list[float] = []
        self.loss_norm_ts: list[float] = []
        self.eval_ts: list[float] = []

        # register the shutdown handler
        atexit.register(self.shutdown)

    # match log args format with level
    def log(self, message: str, *args, level: str = "info") -> None:
        """
        Log a message.

        Args:
            message (str): The message.
            *args: The arguments.
            level (str): The log level.

        Returns:
            None
        """
        self.logger.log(
            getattr(logging, level.upper()),
            f"rank={self.global_rank}.{self.local_rank}/{self.world_size}: {message}",
            *args,
        )

    def shutdown(self) -> None:
        """
        Clean up the files and pools at shutdown.

        Returns:
            None
        """
        # close the log file
        try:
            self.log_file.close()
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error closing log file: %s", str(e))

        # close the eval file
        try:
            self.eval_file.close()
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error closing eval file: %s", str(e))

        # close the object log file
        try:
            self.object_log_file.close()
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error closing object log file: %s", str(e))

        try:
            self.sample_thread_pool.shutdown(wait=True, cancel_futures=True)
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error closing thread pool: %s", str(e))

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the trainer.
        """
        self.shutdown()

    @staticmethod
    def load_training_config(config_path: Path) -> dict:
        """
        Load the training configuration.

        Args:
            config_path (Path): The path to the training configuration.

        Returns:
            dict: The training configuration.
        """
        return json.loads((config_path / "training.json").read_text())

    def get_tokenizer(self, tokenizer_name: str) -> PreTrainedTokenizerFast:
        """
        Get the tokenizer.

        Args:
            tokenizer_name (str): The name of the tokenizer.

        Returns:
            PreTrainedTokenizerFast: The tokenizer.
        """
        self.log("Loading tokenizer %s", tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.log(
            "Loaded tokenizer %s with vocab size=%d", tokenizer_name, len(tokenizer)
        )
        return tokenizer

    @abc.abstractmethod
    def setup_model(
        self,
        tokenizer: PreTrainedTokenizerFast,
        precision: Optional[torch.dtype] = None,
    ) -> None:
        """
        Load the model.

        Returns:
            torch.nn.Module: The model.
        """

    def setup_optimizer(
        self,
    ) -> None:
        """
        Get the optimizer, which defaults to AdamW.

        Args:

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),  # type: ignore
            lr=self.training_config.get("optimizer", {}).get("peak_lr", DEFAULT_LR),
            fused=self.training_config.get("optimizer", {}).get("fused", True),
        )
        self.log("Initialized optimizer: %s", self.optimizer)

    def setup_scheduler(
        self,
    ) -> None:
        """
        Get the scheduler, which defaults to our basic trapezoidal generalization.

        Args:

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The scheduler.
        """
        optimizer_config = self.training_config.get("optimizer", {})
        scheduler_type = optimizer_config.get(
            "scheduler_type", "linear_warmup_cooldown"
        )

        if scheduler_type in ("linear", "linear_warmup_cooldown"):
            self.scheduler = LinearWarmupCooldownScheduler(
                self.optimizer,
                peak_lr=optimizer_config.get("peak_lr", DEFAULT_LR),
                start_lr=optimizer_config.get("start_lr", None),
                end_lr=optimizer_config.get("end_lr", None),
                warmup_steps=optimizer_config.get("warmup_steps", DEFAULT_WARMUP_STEPS),
                peak_steps=optimizer_config.get(
                    "peak_steps", DEFAULT_TOTAL_STEPS // 2
                ),  # default to half of total steps
                total_steps=optimizer_config.get("total_steps", DEFAULT_TOTAL_STEPS),
            )
        elif scheduler_type in ("exponential", "exponential_warmup_cooldown"):
            self.scheduler = ExponentialWarmupCooldownScheduler(
                self.optimizer,
                start_lr=optimizer_config.get("start_lr", DEFAULT_LR),
                warmup_factor=optimizer_config.get("warmup_factor", 2.0),
                warmup_steps_per_period=optimizer_config.get(
                    "warmup_steps_per_period", 100
                ),
                warmup_periods=optimizer_config.get("warmup_periods", 8),
                constant_steps=optimizer_config.get("constant_steps", 100),
                cooldown_factor=optimizer_config.get("cooldown_factor", 2.0),
                cooldown_steps_per_period=optimizer_config.get(
                    "cooldown_steps_per_period", 100
                ),
                cooldown_periods=optimizer_config.get("cooldown_periods", 8),
                last_epoch=-1,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        self.log("Initialized scheduler: %s", self.scheduler)

    def get_lr(self) -> float:
        """
        Get the current learning rate.

        Returns:
            float: The learning rate.
        """
        return self.scheduler.get_last_lr()[0]  # type: ignore

    def log_step(self) -> None:
        """
        Log the current entry to internal and external file log.
        """
        # add time if not there
        if "time" not in self.step_entry:
            self.step_entry["time"] = datetime.datetime.now().isoformat()

        # offload the 'objects' entry if provided into a sidecar; otherwise, this blows up the time to calculate
        # or load the model
        entry_objects = self.step_entry.pop("objects", None)
        if entry_objects:
            self.object_log_file.write(
                json.dumps(  # type: ignore
                    {
                        "step": self.step_entry.get("step", 0),
                        "objects": entry_objects,
                    },
                    default=str,
                )
                + "\n"
            )
            self.object_log_file.flush()

        # add to list
        self.step_logs.append(self.step_entry)
        self.log_file.write(json.dumps(self.step_entry, default=str) + "\n")
        self.step_entry.clear()
        self.log_file.flush()

    def get_trailing_loss(self, steps: int = 10) -> float:
        """
        Get the trailing loss.

        Args:
            steps (int): The number of steps to look back.

        Returns:
            float: The trailing loss.
        """
        losses = [loss for loss in self.loss_ts[-steps:] if loss is not None]
        return sum(losses) / len(losses) if losses else None  # type: ignore

    def forward(self, **inputs: Any) -> MaskedLMOutput:
        """
        Forward pass of the model.

        Args:
            inputs (Any): The inputs.

        Returns:
            dict: The model outputs.
        """
        # track timing
        start_time = time.time()

        # log begin
        self.log("Beginning forward pass")

        # get the model outputs
        output = self.model.forward(**inputs)  # type: ignore

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
        Backward pass of the model.

        Args:
            loss (torch.Tensor): The loss.
        """
        # log it
        self.log("Beginning backward pass with loss=%0.2f", loss)

        # track timing
        start_time = time.time()

        # backward pass with kwargs
        loss.backward(**kwargs)

        # track timing
        end_time = time.time()

        # log it
        self.log("Completed backward pass in %0.2f seconds", end_time - start_time)

        self.step_entry["backward_time"] = end_time - start_time

    def step(self) -> None:
        """
        Step the optimizer.

        Returns:
            None
        """
        # track timing
        start_time = time.time()

        # log
        self.log("Beginning optimizer step")

        # step the optimizer
        self.optimizer.step()  # type: ignore
        self.scheduler.step()  # type: ignore

        # zero the optimizer
        self.optimizer.zero_grad()  # type: ignore

        # track timing
        end_time = time.time()

        # log it
        self.log("Completed optimizer step in %0.2f seconds", end_time - start_time)

        self.step_entry["optimizer_time"] = end_time - start_time

    def save(self) -> None:
        """
        Save the model.
        """
        # save the model
        self.log("Saving model to %s", self.checkpoint_path)

        try:
            self.model.save_pretrained(self.checkpoint_path)  # type: ignore
        except Exception as e:  # pylint: disable=broad-except
            # check for `safe_serialization=False` in the str
            if "safe_serialization=False" in str(e):
                self.log(
                    "Falling back to torch.bin format due to safe_serialization=False"
                )
                self.model.save_pretrained(  #  type: ignore
                    self.checkpoint_path, safe_serialization=False
                )  # type: ignore

        self.log("Saving tokenizer to %s", self.checkpoint_path)
        self.tokenizer.save_pretrained(self.checkpoint_path)  # type: ignore

    @abc.abstractmethod
    def get_sample(self, device: Optional[str] = None) -> dict[str, torch.Tensor]:
        """
        Get a sample for training.

        Args:
            device (str): The device

        Returns:
            Any: The sample.
        """

    @abc.abstractmethod
    def load_eval_data(self, num_samples: int) -> None:
        """
        Load the evaluation data.

        Args:
            num_samples (int): The number of samples to load.
        """

    def populate_queue(self) -> None:
        """
        Populate the sample queue.
        """
        try:
            self.sample_queue.put(self.get_sample(), timeout=30.0)
        except Exception as e:  # pylint: disable=broad-except
            self.log("Error adding sample to queue: %s", str(e), level="error")

    def get_next_sample(self) -> dict[str, torch.Tensor]:
        """
        Get the next sample from the queue.

        Returns:
            dict[str, torch.Tensor]: The sample.
        """
        while True:
            try:
                self.log("Queue size: %d", self.sample_queue.qsize())
                sample = self.sample_queue.get(timeout=1)
            except queue.Empty:
                self.log(
                    "Sample queue empty, hitting get_sample() directly", level="warning"
                )

                # get next sample from queue
                sample = self.get_sample()
            except Exception as e:  # pylint: disable=broad-except
                # sleep and try again
                self.log("Error getting sample from queue: %s", str(e), level="error")
                time.sleep(DEFAULT_SAMPLE_SLEEP)
                continue

            # now move onto the right device
            try:
                for key in sample:
                    if isinstance(sample[key], torch.Tensor):
                        if sample[key].device != self.device:
                            sample[key] = sample[key].to(self.device)
            except Exception as e:  # pylint: disable=broad-except
                self.log("Error moving sample to device: %s", str(e), level="error")

            return sample

    def load_state(self, checkpoint_path: Path) -> dict[str, int]:
        """
        Load the model state from a checkpoint.

        Args:
            checkpoint_path (Path): The path to the checkpoint.
        """
        # try to load the log.jsonl file if it exists;
        # get the final step and epoch if it does
        step = 0
        epoch = 0
        log_file_path = checkpoint_path / "log.jsonl"
        if checkpoint_path.exists() and log_file_path.exists():
            self.log("Loading log file %s", log_file_path)
            try:
                with log_file_path.open("rt", encoding="utf-8") as log_file:
                    for line in log_file:
                        # load and pop out the objects to avoid OOM on low ram, long step runs
                        entry = json.loads(line)
                        entry.pop("objects", None)
                        self.step_logs.append(entry)
                        self.loss_ts.append(entry["loss"])

                # get the step and epoch from the final entry
                if len(self.step_logs) > 0:
                    step = self.step_logs[-1]["step"] + 1
                    epoch = self.step_logs[-1]["epoch"]
                    self.log(
                        "Reloaded state from log file: step=%d, epoch=%d", step, epoch
                    )
                else:
                    self.log(
                        "No entries in log file, starting from scratch @ step=0, epoch=0"
                    )
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error loading log file: {str(e)}")

        return {"step": step, "epoch": epoch}

    def eval(self) -> dict[str, Any]:
        """
        Run an eval loop by calculating the loss distribution on the eval samples.

        Disable the gradient calculation during this phase.

        Returns:
            dict: The evaluation results.
        """
        # log
        self.log("Running eval loop")

        # set the model to eval mode
        self.model.eval()  # type: ignore

        # disable gradient calculation
        sample_loss = []
        with torch.no_grad():
            # get the eval loss distribution
            for sample in self.eval_samples:
                # move onto device
                for key in sample:
                    if isinstance(sample[key], torch.Tensor):
                        sample[key] = sample[key].to(self.device)
                outputs = self.forward(**sample)
                sample_loss.append(outputs.loss.item())

            # get the state dict on cpu efficiently for svd calcs
            layer_stats = {
                key: get_layer_svd_stats(value.cpu())
                for key, value in self.model.state_dict().items()  # type: ignore
                if len(value.shape) == 2 and key.endswith("intermediate.dense.weight")
            }

            # get the mean and median of the svd values for the intermediate layers
            mean_svds = [
                float(layer_stats[layer_name]["mean_ratio_1"])
                for layer_name in layer_stats
            ]

            median_svds = [
                float(layer_stats[layer_name]["median_ratio_1"])
                for layer_name in layer_stats
            ]

            # get 5th and 95th percentiles
            quantiles = statistics.quantiles(sample_loss, n=100)

            # get stats
            eval_stats = {
                "mean": statistics.mean(sample_loss),
                "median": statistics.median(sample_loss),
                "std": statistics.stdev(sample_loss),
                "min": min(sample_loss),
                "p5": quantiles[5],
                "p95": quantiles[95],
                "max": max(sample_loss),
                "num_samples": len(sample_loss),
                "svd_mean_ratio_1": statistics.mean(mean_svds),
                "svd_median_ratio_1": statistics.mean(median_svds),
            }

        # set the model back to train mode
        self.model.train()  # type: ignore

        # log it
        self.log("Eval stats: %s", json.dumps(eval_stats, default=str, indent=2))

        # write to eval file
        self.eval_file.write(
            json.dumps(
                {
                    "step": self.step_entry.get("step", 0),
                    **eval_stats,
                },
                default=str,
            )
            + "\n"
        )
        self.eval_file.flush()

        return eval_stats

    def get_grad_norm(self) -> float:
        """
        Safely get the gradient norm.

        Returns:
            float: The gradient norm.
        """
        try:
            grad_tensors = torch.stack(
                [
                    param.grad.detach().norm()
                    for param in self.model.parameters()  # type: ignore
                    if param.grad is not None
                ]
            )
            grad_norm = torch.norm(grad_tensors).item()
        except Exception:  # pylint: disable=broad-except
            grad_norm = 0.0

        return grad_norm

    def clip_grad(self, grad_norm: float) -> bool:
        """
        Clip the gradient.

        Args:
            grad_norm (float): The gradient norm.

        Returns:
            bool: Whether the gradient was clipped.
        """
        # get the relevant params
        optimizer_config = self.training_config.get("optimizer", {})
        max_grad_norm = optimizer_config.get("max_grad_norm", DEFAULT_MAX_GRAD_NORM)
        max_grad_norm_pct = optimizer_config.get(
            "max_grad_norm_pct", DEFAULT_MAX_GRAD_NORM_PCT
        )

        if max_grad_norm:
            if not max_grad_norm_pct:
                clip_threshold = max_grad_norm
            else:
                if len(self.loss_norm_ts) < 100:
                    clip_threshold = max_grad_norm
                else:
                    clip_threshold = statistics.quantiles(self.loss_norm_ts, n=100)[
                        max_grad_norm_pct
                    ]

            # log it
            self.log("Clipping gradients at %0.2f", clip_threshold)
            self.step_entry["clip_threshold"] = clip_threshold

            # clip based on threshold if we're over
            if grad_norm > clip_threshold:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),  # type: ignore
                    clip_threshold,
                )
                return True

        return False

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

        # populate the eval sample
        print("Populating eval samples...")
        self.load_eval_data(num_eval_samples)
        print(f"Populated {len(self.eval_samples)} eval samples.")

        # start training loop
        train_start_time = time.time()
        train_status = False

        # compile the whole model here
        self.log("Beginning model compilation...")
        self.model = torch.compile(self.model, fullgraph=True).to(
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
                            sample = self.get_next_sample()
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
                    if (step + 1) % gradient_accumulation_steps == 0:
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
                    prog_bar.set_postfix(
                        {
                            "loss": f"{self.step_entry.get("loss", 99.9):0.2f}",
                            "loss_100": f"{self.get_trailing_loss(100):0.3f}",
                            "loss_1000": f"{self.get_trailing_loss(1000):0.3f}",
                            "last_eval": f"{last_eval:0.2f}",
                            "grad_norm": f"{grad_norm:0.2f}",
                            "lr": f"{self.get_lr():1.1e}",
                            "step_time": f"{self.step_entry['step_time']:0.2f}",
                            "token_rate": f"{token_rate:0.2f}",
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
                print(f"Error during training: {e}")
                traceback.print_exc()
                self.log("Error during training: %s", str(e), level="error")
                self.log(traceback.format_exc(), level="error")
            finally:
                # final save
                self.save()

                # force thread pool shutdown after saving
                try:
                    self.sample_thread_pool.shutdown(wait=False)
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error shutting down sample thread pool: {e}")

        return train_status
