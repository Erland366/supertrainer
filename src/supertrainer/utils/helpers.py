# MIT License
#
# Copyright (c) 2024 Edd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from huggingface_hub import login

import wandb

from .. import types
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
    dataset_name_or_path: str | os.PathLike | types.Dataset,  # noqa: F821 # type: ignore
    tokenizer_name_or_path: str | os.PathLike,
    set: str = "train",
    is_chat_formatted: bool = False,
    chat_template: str = "{instr} {inp} {out}",
) -> int:
    """
    Finds the maximum number of tokens in a dataset.

    Args:
        dataset_name_or_path (str | os.PathLike | types.Dataset):
            The name or path of the dataset, or a DatasetDict or Dataset object.
        tokenizer_name_or_path (str | os.PathLike): The name or path of the tokenizer.
        set (str, optional): The dataset split to use. Defaults to "train".
        is_chat_formatted (bool, optional): Whether the dataset is chat-formatted.
            Defaults to False.
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
            "You put DatasetDict but did not specify the argument 'set'. "
            "Assuming you're going to use the 'train' dataset."
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


def clean_gpu_cache(model: Any | None = None):
    """
    Cleans the GPU cache by deleting the model object and emptying the CUDA cache.

    Args:
        model (Any | None, optional): The model object to be deleted. Defaults to None.

    Returns:
        None

    Raises:
        None

    """
    import gc

    import torch

    if model is not None:
        try:
            del model
        except Exception:
            pass

    for _ in range(10):
        torch.cuda.empty_cache()
        gc.collect()

    print("GPU cache cleaned")


def split_train_test_validation(
    dataset: types.Dataset, select_subset: int | float = 0
) -> types.Dataset:
    """
    Splits the dataset into train, validation, and test subsets.
    Args:
        dataset (types.Dataset): The dataset to be split.
        select_subset (int | float, optional): The size of the subset to select.
            If float, it represents the percentage of the dataset to select.
            If int, it represents the number of samples to select.
            Defaults to 0, which means no subset will be selected.
    Returns:
        types.Dataset: A DatasetDict containing the train, validation, and test subsets.
    """
    from datasets import DatasetDict

    if isinstance(dataset, DatasetDict):
        logger.warning("Dataset is a DatasetDict. We will you're going to use the 'train' dataset.")
        dataset = dataset["train"]

    if select_subset > 0:
        if isinstance(select_subset, float):
            select_subset = int(len(dataset) * select_subset)
        dataset = dataset.select(range(select_subset))

    # shuffle dataset
    dataset = dataset.shuffle(seed=42)

    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    train_valid_split = train_dataset.train_test_split(test_size=0.2)
    train_dataset = train_valid_split["train"]
    valid_dataset = train_valid_split["test"]

    split_dataset = DatasetDict(
        {"train": train_dataset, "validation": valid_dataset, "test": test_dataset}
    )

    return split_dataset
