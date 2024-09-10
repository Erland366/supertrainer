from __future__ import annotations

import os
from importlib import import_module
from typing import Any

import wandb
from huggingface_hub import login

from .logger import logger


def login_hf(environ_name: str = "HUGGINGFACE_API_KEY", token: str | None = None):
    if token is None:
        token = os.getenv(environ_name)
        logger.debug(f"Use token from environment variable {environ_name}")
    login(token=token)


def login_wandb(environ_name: str = "WANDB_API_KEY", token: str | None = None):
    if token is None:
        token = os.getenv(environ_name)
        logger.debug(f"Use token from environment variable {environ_name}")
    wandb.login(key=token)


def is_flash_attention_available() -> bool:
    import importlib

    HAS_FLASH_ATTENTION = importlib.util.find_spec("flash_attn")
    return HAS_FLASH_ATTENTION is not None


def memory_stats():
    """
    Gather memory statistics related to GPU usage.
    """
    import torch

    logger.debug("Gathering memory statistics")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.debug(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.debug(f"{start_gpu_memory} GB of memory reserved.")
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")


def find_max_tokens(
    dataset_name_or_path: str | os.PathLike | "DatasetDict" | "Dataset",  # noqa: F821 # type: ignore
    tokenizer_name_or_path: str | os.PathLike,
    set: str = "train",
    is_chat_formatted: bool = False,
    chat_template: str = "{instr} {inp} {out}",
) -> int:
    """
    Finds the maximum number of tokens in a dataset.

    Args:
        dataset_name_or_path (str | os.PathLike | "DatasetDict" | "Dataset"): The name or path of the dataset, or a DatasetDict or Dataset object.
        tokenizer_name_or_path (str | os.PathLike): The name or path of the tokenizer.
        set (str, optional): The dataset split to use. Defaults to "train".
        is_chat_formatted (bool, optional): Whether the dataset is chat-formatted. Defaults to False.
        chat_template (str, optional): The chat template. Defaults to "{instr} {inp} {out}".

    Returns:
        int: The maximum number of tokens in the dataset.
    """

    from datasets import DatasetDict, load_dataset
    from transformers import AutoTokenizer

    if isinstance(dataset_name_or_path, (str, os.PathLike)):
        dataset = load_dataset(dataset_name_or_path)
    else:
        dataset = dataset_name_or_path

    if isinstance(dataset, DatasetDict):
        logger.debug(
            "You put DatasetDict but did not specify the argument 'set'. Assuming you're going to use the 'train' dataset."
        )
        dataset = dataset[set]

    logger.debug(f"Dataset columns: {dataset.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def process_data(
        examples: dict, tokenizer: "AutoTokenizer", is_chat_formatted: bool = True
    ) -> list[int]:
        if is_chat_formatted:
            texts = examples["text"]
        else:
            texts = [
                chat_template.format(instr=instr, inp=inp, out=out)
                for instr, inp, out in zip(
                    examples["instruction"], examples["input"], examples["output"]
                )
            ]

        return [len(tokenizer.tokenize(text)) for text in texts]

    token_counts = dataset.map(
        lambda x: {"token_count": process_data(x, tokenizer, is_chat_formatted)},
        batched=True,
        remove_columns=dataset.column_names,
    )

    return max(token_counts["token_count"])


def print_and_log(message: str, log_level: str = "info", depth: int = 1):
    from .logger import logger

    print(message)

    # Log the message using loguru
    log_func = getattr(logger.opt(depth=depth), log_level, None)
    if log_func is None:
        raise ValueError(f"Invalid log level: {log_level}")

    log_func(f"{message}")


def import_class(class_path: str) -> Any:
    """
    Dynamically import a class from a string.

    Args:
        class_path (str): The full path to the class, e.g.,
                          "package.module.ClassName"

    Returns:
        Any: The imported class

    Raises:
        ImportError: If the module or class cannot be imported
        AttributeError: If the class is not found in the module
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Error importing {class_path}: {str(e)}")
