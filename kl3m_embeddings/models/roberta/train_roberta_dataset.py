"""
Train a Roberta model with the HuggingFace Trainer API
"""

# imports
import math
import random
from collections import deque, Counter

# packages
from huggingface_hub import hf_api
from datasets import (
    load_dataset,
    Dataset,
    interleave_datasets,
    concatenate_datasets,
)
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

# project


INPUT_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
    "alea-institute/kl3m-004-128k-cased-mlm"
)
CONTEXT_WINDOW = 512
SPECIAL_TOKENS = 3
RECORD_SIZE = CONTEXT_WINDOW - SPECIAL_TOKENS


def list_kl3m_datasets() -> list[str]:
    # kl3m datasets
    kl3m_datasets = list(hf_api.list_datasets(author="alea-institute"))
    return kl3m_datasets


KL3M_DATASETS = tuple(list_kl3m_datasets())

EVAL_DATASETS = ("alea-institute/kl3m-data-eval-10k-001",)


# trapezoidal learning rate scheduler
class TrapezoidalCosineScheduler(LRScheduler):
    """
    Trapezoidal LR scheduler that goes like:
    - linear warmup from min_lr to max_lr over warmup_steps
    - max_lr for plateau_steps + cosine term with period cosine_period
    - 1 - sqrt decay from max_lr to min_lr over decay_steps
    """

    def __init__(
        self,
        optimizer: Optimizer,
        start_lr: float,
        plateau_lr: float,
        end_lr: float,
        warmup_steps: int,
        plateau_steps: int,
        cooldown_steps: int,
        cosine_period: int,
        cosine_magnitude: float,
    ) -> None:
        self.start_lr = start_lr
        self.plateau_lr = plateau_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        self.plateau_steps = plateau_steps
        self.cooldown_steps = cooldown_steps
        self.cosine_period = cosine_period
        self.cosine_magnitude = cosine_magnitude

        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        # get the current step
        step = self.last_epoch

        if step < self.warmup_steps:
            # linear warmup
            return [
                self.start_lr
                + (self.plateau_lr - self.start_lr) * step / self.warmup_steps
            ]
        elif step < self.warmup_steps + self.plateau_steps:
            # plateau with cosine term
            cos_term = (
                self.plateau_lr
                * self.cosine_magnitude
                * math.sin(math.pi * (step - self.warmup_steps) / self.cosine_period)
            )
            return [self.plateau_lr + cos_term]
        else:
            # 1 - sqrt cooldown
            cooldown_step_index = step - (self.warmup_steps + self.plateau_steps)
            return [
                self.plateau_lr
                - (self.plateau_lr - self.end_lr)
                * math.sqrt(cooldown_step_index / self.cooldown_steps)
            ]


def token_entropy(tokens: list[int]) -> float:
    """
    Calculate the entropy of a list of tokens.
    """
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    entropy = sum(
        -count / total_tokens * math.log2(count / total_tokens)
        for count in token_counts.values()
    )
    return entropy


def merge_token_records(
    records: list[dict],
    record_size: int = RECORD_SIZE,
    sep_token_id=INPUT_TOKENIZER.sep_token_id,
    mime_types: tuple[str] = ("text/markdown", "text/plain"),
) -> list[list[int]]:
    """
    Fill record size by combining sequences with sep token using queue of reversed tokens.

    Args:
        records (list[dict]): The records.
        record_size (int): The record size.
        sep_token_id (int): The separator token id.
        mime_types (tuple[str]): The mime types to include.

    Returns:
        list[list[int]]: The merged token records.
    """
    output_records: list[list[int]] = []
    pending_record: list[int] = [INPUT_TOKENIZER.cls_token_id]

    for record_index in range(len(records["tokens"])):
        # only keep text/plain, text/markdown
        record_mime_type = records["mime_type"][record_index]
        if record_mime_type not in mime_types:
            continue

        # get the record tokens
        record_tokens = records["tokens"][record_index]

        # check if the record ratio is weird
        try:
            # record_tokens = record.get("tokens", [])
            content = INPUT_TOKENIZER.decode(record_tokens)
            num_bytes_non_ws = sum(1 for c in content if not c.isspace())
            non_ws_bytes_per_token = num_bytes_non_ws / len(record_tokens)

            if non_ws_bytes_per_token < 2.5:
                continue

            # create a queue of reversed tokens
            token_queue = deque(reversed(record_tokens))

            # fill the pending record
            while token_queue:
                if len(pending_record) < record_size - 1:
                    pending_record.append(token_queue.pop())
                else:
                    pending_record.append(sep_token_id)
                    output_records.append(pending_record)
                    pending_record = [INPUT_TOKENIZER.cls_token_id]

            # add an eos token to the pending record
            if len(pending_record) < record_size - 1:
                pending_record.append(INPUT_TOKENIZER.eos_token_id)
            else:
                output_records.append(pending_record)
                pending_record = [INPUT_TOKENIZER.cls_token_id]
        except Exception as e:
            print(f"Failed to decode record: {e}")
            continue

    # final shuffle
    random.shuffle(output_records)

    return output_records


# map the tokens column to text with the INPUT_TOKENIZER
def decode_dataset_tokens(records: dict, record_size: int = RECORD_SIZE) -> dict:
    """
    Decode the tokens in the dataset.

    Args:
        records (dict): The records.
        record_size (int): The record size.

    Returns:
        dict: The example with decoded tokens.
    """
    output_records = merge_token_records(records, record_size=record_size)

    counts = {}
    for record in output_records:
        counts[len(record)] = counts.get(len(record), 0) + 1
    assert max(counts.keys()) <= record_size, (
        f"Record size exceeded: {max(counts.keys())}"
    )

    return {"input_ids": output_records}


def load_datasets(streaming: bool = True) -> Dataset:
    """
    Load the datasets.

    Args:
        streaming (bool): Whether to stream the datasets.

    Returns:
        Dataset: The dataset.
    """
    # load the dataset
    datasets = []
    for dataset_result in KL3M_DATASETS:
        # if 'alea-institute/kl3m-data-usc' not in dataset_result.id:
        #    continue
        if (
            not dataset_result.id.startswith("alea-institute/kl3m-data-")
            or "-eval-" in dataset_result.id
        ):
            continue
        datasets.append(
            load_dataset(dataset_result.id, split="train", streaming=streaming)
        )

    # interleave
    combined_dataset = interleave_datasets(datasets, stopping_strategy="all_exhausted")

    # map them all in parallel
    combined_dataset = combined_dataset.map(
        decode_dataset_tokens,
        batched=True,
        remove_columns=combined_dataset.column_names,
    )

    combined_dataset = combined_dataset.shuffle()

    return combined_dataset


def load_eval_datasets(streaming: bool = True) -> Dataset:
    """
    Load the evaluation datasets.

    Args:
        streaming (bool): Whether to stream the datasets.

    Returns:
        Dataset: The dataset.
    """
    # load the dataset
    datasets = []
    for dataset_id in EVAL_DATASETS:
        datasets.append(load_dataset(dataset_id, split="train", streaming=streaming))

    # interleave
    combined_dataset = concatenate_datasets(datasets)

    # map them all in parallel
    combined_dataset = combined_dataset.map(
        decode_dataset_tokens,
        batched=True,
        remove_columns=combined_dataset.column_names,
    )

    # shuffle and take 100
    combined_dataset = combined_dataset.shuffle().take(1000)

    return combined_dataset


if __name__ == "__main__":
    # load the datasets
    train_dataset = load_datasets()
    eval_dataset = load_eval_datasets()

    # set up collator with output tokenizer
    collator = DataCollatorForLanguageModeling(
        tokenizer=INPUT_TOKENIZER,
        mlm=True,
    )

    # try to reload
    try:
        print("Loading model...")
        model = RobertaForMaskedLM.from_pretrained("./roberta-model")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Creating a new model...")

        # create a roberta model from scratch
        hidden_size = 512
        intermediate_size = 2048
        num_hidden_layers = 16
        num_attention_heads = 16

        model_config = RobertaConfig(
            vocab_size=len(INPUT_TOKENIZER.get_vocab()),
            max_position_embeddings=CONTEXT_WINDOW,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            type_vocab_size=1,
            pad_token_id=INPUT_TOKENIZER.pad_token_id,
            bos_token_id=INPUT_TOKENIZER.cls_token_id,
            eos_token_id=INPUT_TOKENIZER.sep_token_id,
        )
        model = RobertaForMaskedLM(model_config)

    # get params
    hidden_size = model.config.hidden_size
    num_hidden_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    model_name = f"roberta-{hidden_size}-{num_hidden_layers}-{num_attention_heads}"
    print(f"Loaded model: {model_name}")

    # get total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params > 1e9:
        model_name = f"roberta-{hidden_size}-{num_hidden_layers}-{num_attention_heads}"
        model_size = f"{total_params / 1e9:0.2f}B"
    else:
        model_name = f"roberta-{hidden_size}-{num_hidden_layers}-{num_attention_heads}"
        model_size = f"{total_params / 1e6:0.2f}M"

    print(f"Total parameters: {model_size}")

    # set up cosine scheduler
    max_steps = 1e6
    warmup_proportion = 0.1
    cooldown_proportion = 0.1
    start_lr = 1e-7
    peak_lr = 1e-4

    # set up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=start_lr,
        betas=(0.9, 0.95),
        weight_decay=1e-2,
    )

    scheduler = TrapezoidalCosineScheduler(
        optimizer,
        start_lr=start_lr,
        plateau_lr=peak_lr,
        end_lr=start_lr / 10.0,
        warmup_steps=int(max_steps * warmup_proportion),
        plateau_steps=int(max_steps * (1 - warmup_proportion - cooldown_proportion)),
        cooldown_steps=int(max_steps * cooldown_proportion),
        cosine_period=100,
        cosine_magnitude=0.1,
    )

    # set up training arguments
    training_args = TrainingArguments(
        run_name=f"{model_name}",
        output_dir="./roberta-model",
        overwrite_output_dir=True,
        # training setup
        max_steps=int(max_steps),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        # max_grad_norm=128.0,
        max_grad_norm=8.0,
        # eval
        eval_on_start=True,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=1000,
        # save model
        save_steps=1000,
        save_safetensors=True,
        save_total_limit=3,
        save_strategy="steps",
        # logging setup
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=1,
    )

    # create the trainer
    trainer = Trainer(
        # model
        model=model,
        # optimizer
        optimizers=(optimizer, scheduler),
        # training args
        args=training_args,
        # collator
        data_collator=collator,
        # datasets
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # train the model
    try:
        trainer.train()

    except KeyboardInterrupt:
        print("Interrupted training...")
    except Exception as e:
        print(f"Failed to train model: {e}")
    finally:
        # save the model
        trainer.save_model("./roberta-model")
        INPUT_TOKENIZER.save_pretrained("./roberta-model")
