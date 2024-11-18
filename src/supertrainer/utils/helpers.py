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
from contextlib import contextmanager
from importlib import import_module
from typing import Any

import psutil
import torch
from huggingface_hub import login, whoami
from packaging import version

import wandb

from .. import type_hinting
from .logger import logger

torch_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float32
)


def remove_config_eval(config: type_hinting.Config) -> type_hinting.Config:
    if config.trainer.training_kwargs.get("eval_strategy", None) is not None:
        with config.allow_modification():
            config.trainer.training_kwargs.eval_strategy = "no"

    # This is must with eval, so delete
    if config.trainer.training_kwargs.get("load_best_model_at_end", None) is not None:
        with config.allow_modification():
            del config.trainer.training_kwargs.load_best_model_at_end

    if config.trainer.training_kwargs.get("eval_on_start", None) is not None:
        with config.allow_modification():
            del config.trainer.training_kwargs.eval_on_start

    if config.trainer.training_kwargs.get("do_eval", None) is not None:
        with config.allow_modification():
            config.trainer.training_kwargs.do_eval = False


def login_hf(environ_name: str = "HUGGINGFACE_API_KEY", token: str | None = None):
    if token is None:
        token = os.getenv(environ_name)
        logger.debug(f"Use token from environment variable {environ_name}")
    login(token=token)
    print(f"Hugging Face user: {whoami()['name']}, full name: {whoami()['fullname']}")


def login_wandb(environ_name: str = "WANDB_API_KEY", token: str | None = None, **kwargs):
    if token is None:
        token = os.getenv(environ_name)
        logger.debug(f"Use token from environment variable {environ_name}")
    wandb.login(key=token, **kwargs)


def set_global_seed(seed: int = 42):
    from transformers import set_seed

    set_seed(seed)


def is_flash_attention_available() -> bool:
    import importlib

    HAS_FLASH_ATTENTION = importlib.util.find_spec("flash_attn")
    return HAS_FLASH_ATTENTION is not None


def check_flash_attention_2_support() -> bool:
    """
    Check if the current system supports Flash Attention 2.

    Returns:
        Tuple[bool, str]: (is_supported, reason)
        - is_supported: Boolean indicating if Flash Attention 2 is supported
        - reason: String explaining why it's not supported (if applicable)
    """
    if not torch.cuda.is_available():
        return False

    # Check CUDA version (needs 11.6 or higher)
    cuda_version = torch.version.cuda
    if cuda_version is None:
        return False

    if version.parse(cuda_version) < version.parse("11.6"):
        return False

    # Check GPU architecture (needs Ampere or newer: compute capability >= 8.0)
    cc_major = torch.cuda.get_device_capability()[0]

    if cc_major < 8:
        return False

    return True


def load_model_with_adaptive_attention(model_loader_func, *args, **kwargs):
    if kwargs.get("_attn_implementation") is not None:
        attn_implementation = kwargs.pop("_attn_implementation")
    else:
        attn_implementation = "flash_attention_2" if check_flash_attention_2_support() else "sdpa"
    try:
        model = model_loader_func(*args, _attn_implementation=attn_implementation, **kwargs)
    except ValueError as e:
        error_message = str(e)
        if (
            "does not support an attention implementation through torch.nn.\
            functional.scaled_dot_product_attention yet."
            in error_message
        ):
            attn_implementation = "eager"
            model = model_loader_func(*args, _attn_implementation=attn_implementation, **kwargs)
        else:
            raise
    return model


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
    dataset_name_or_path: str | os.PathLike | type_hinting.Dataset,  # noqa: F821 # type: ignore
    tokenizer_name_or_path: str | os.PathLike,
    set: str = "train",
    is_chat_formatted: bool = False,
    chat_template: str = "{instr} {inp} {out}",
) -> int:
    """
    Finds the maximum number of tokens in a dataset.

    Args:
        dataset_name_or_path (str | os.PathLike | type_hinting.Dataset):
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
    dataset: type_hinting.Dataset, select_subset: int | float = 0
) -> type_hinting.Dataset:
    """
    Splits the dataset into train, validation, and test subsets.
    Args:
        dataset (type_hinting.Dataset): The dataset to be split.
        select_subset (int | float, optional): The size of the subset to select.
            If float, it represents the percentage of the dataset to select.
            If int, it represents the number of samples to select.
            Defaults to 0, which means no subset will be selected.
    Returns:
        type_hinting.Dataset: A DatasetDict containing the train, validation, and test subsets.
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


def convert_size_to_mb(size: int) -> float:
    """
    Convert a size from bytes to megabytes.

    Args:
        size (int): The size in bytes.

    Returns:
        float: The size in megabytes.
    """
    return size / 1024 / 1024


class MemoryTracker:
    def __init__(self) -> None:
        self.max_ram_usage = 0
        self.max_vram_usage = 0
        self.torch_available = False
        try:
            import torch

            self.torch_available = torch.cuda.is_available()
        except ImportError:
            logger.warning(
                "Torch is not available. VRAM tracking will not be available.",
            )

    @contextmanager
    def track_memory(self, description: str):
        process = psutil.Process()
        mem_info_start = process.memory_info()

        if self.torch_available:
            import torch

            gpu_mem_info_start = (
                convert_size_to_mb(torch.cuda.memory_allocated())
                if torch.cuda.is_available()
                else 0
            )
            gpu_mem_reserved_start = (
                convert_size_to_mb(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
            )

        yield

        mem_info_end = convert_size_to_mb(process.memory_info().rss - mem_info_start.rss)

        if self.torch_available:
            gpu_mem_info_end = (
                convert_size_to_mb(torch.cuda.memory_allocated())
                if torch.cuda.is_available()
                else 0
            ) - gpu_mem_info_start
            gpu_mem_reserved_end = (
                convert_size_to_mb(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
            ) - gpu_mem_reserved_start

        self.max_ram_usage = max(self.max_ram_usage, mem_info_end)
        self.max_vram_usage = max(self.max_vram_usage, gpu_mem_info_end)

        print_and_log(
            f"{description} - RAM Usage: RSS: {mem_info_end:.2f} MB",
            depth=1,
        )
        if self.torch_available:
            print_and_log(
                f"{description} - VRAM Usage: Allocated: {gpu_mem_info_end:.2f} MB, "
                f"Reserved: {gpu_mem_reserved_end:.2f} MB",
                depth=1,
            )
        else:
            print_and_log(f"{description} - CUDA is not available.", depth=1)

    def print_summary(self):
        print_and_log(f"Maximum RAM Usage: {self.max_ram_usage:.2f} MB", depth=1)
        if self.torch_available:
            print_and_log(f"Maximum VRAM Usage: {self.max_vram_usage:.2f} MB", depth=1)


def get_model_name(model: Any) -> str:
    """
    Get the name of the model.

    Args:
        model (Any): The model object.

    Returns:
        str: The name of the model.
    """
    try:
        name = model.config.architectures[0]
    except AttributeError:
        name = model.__class__.__name__
    return name


def load_dataset_plus_plus(*args, **kwargs) -> "Dataset":  # noqa: F821 # type: ignore
    """
    Load a dataset from the specified path.

    This function attempts to load a dataset using the `load_dataset`
    function from the `datasets` library.
    If a `ValueError` is raised indicating that the dataset was saved using
    `save_to_disk`, it will instead
    load the dataset using the `load_from_disk` method.

    Args:
        path (str): The path to the dataset.

    Returns:
        Dataset: The loaded dataset.

    Raises:
        ValueError: If the dataset cannot be loaded using either `load_dataset` or `load_from_disk`.
    """
    from datasets import Dataset, DatasetDict, load_dataset

    if kwargs.get("subsets", None) is not None:
        subsets = kwargs.pop("subsets")
    else:
        subsets = None

    try:
        if subsets is not None:
            dataset_subsets = {}
            for subset in subsets:
                dataset_subsets[subset] = load_dataset(*args, name=subset, **kwargs)
            dataset = DatasetDict(dataset_subsets)
        else:
            dataset = load_dataset(*args, **kwargs)
    except ValueError as e:
        if str(e) == (
            "You are trying to load a dataset that was saved using "
            "`save_to_disk`. Please use `load_from_disk` instead."
        ):
            # take path from args or kwargs
            # if path is provided, put it as dataset_path instead
            if kwargs.get("path"):
                kwargs["dataset_path"] = kwargs.pop("path")

            dataset = Dataset.load_from_disk(*args, **kwargs)
        else:
            raise e
    return dataset
