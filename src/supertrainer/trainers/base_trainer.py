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

import datetime
import json
import os
from abc import ABC, abstractmethod

import torch
from huggingface_hub import HfApi, create_repo
from transformers import BitsAndBytesConfig

import wandb
from supertrainer import logger, types
from supertrainer.utils import memory_stats


class ABCTrainer(ABC):
    _model: "AutoModelForCausalLM" | "FastLanguageModel" | None = None  # noqa # type:ignore
    _tokenizer: "AutoTokenizer" | None = None  # noqa # type: ignore
    config: types.Config
    dataset: types.Dataset  # noqa # type: ignore

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self):
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> "AutoTokenizer":  # noqa # type: ignore
        pass

    @property
    @abstractmethod
    def model(self) -> "AutoModelForCausalLM" | "FastLanguageModel" | None:  # noqa # type: ignore
        pass

    @abstractmethod
    def postprocess_config(self, config: types.Config) -> types.Config:
        return config


class BaseTrainer(ABCTrainer):
    def __init__(self, config: types.Config, dataset: types.Dataset) -> None:
        self.dataset = dataset
        self.config = config

    def train(self):
        raise NotImplementedError

    def postprocess_config(self, config: types.Config) -> types.Config:
        # TODO: Move it from here since it's not modular enough
        with config.allow_modification():
            config.trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config.trainer.training_kwargs.fp16 = not torch.cuda.is_bf16_supported()
            config.trainer.training_kwargs.bf16 = torch.cuda.is_bf16_supported()
            config.trainer.training_kwargs.hub_model_id = (
                config.trainer.training_kwargs.hub_model_id
                + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            )

            # mainly for data type casting purpose
            config.trainer.dtype = (
                torch.bfloat16 if config.trainer.training_kwargs.bf16 else torch.float36
            )

            # Construct the run_name
            model_name = config.model_name.split("/")[-1]
            dataset_name = config.dataset.dataset_kwargs.path.split("/")[-1]
            lora_rank = config.trainer.peft_kwargs.r
            learning_rate = config.trainer.training_kwargs.learning_rate
            num_epochs = config.trainer.training_kwargs.num_train_epochs

            run_name = (
                f"{model_name}_{dataset_name}_r{lora_rank}_lr{learning_rate}_e{num_epochs}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            config.trainer.training_kwargs.run_name = run_name

            # output_dir add run_name
            config.trainer.training_kwargs.output_dir = os.path.join(
                config.trainer.training_kwargs.output_dir, run_name
            )

            # TRAINING STUFF HERE
            config.trainer.bitsandbytes_kwargs.bnb_4bit_compute_dtype = (
                config.trainer.bitsandbytes_kwargs.bnb_4bit_compute_dtype or "bfloat16"
                if torch.cuda.is_bf16_supported()
                else "float16"
            )
            quantization_config = BitsAndBytesConfig(**config.trainer.bitsandbytes_kwargs)

            config.trainer.model_kwargs.device_map = config.trainer.model_kwargs.device_map or {
                "": torch.cuda.current_device() if torch.cuda.is_available() else None
            }
            config.trainer.model_kwargs.attn_implementation = (
                config.trainer.model_kwargs.attn_implementation or "sdpa"
            )
            config.trainer.model_kwargs.quantization_config = quantization_config

        logger.debug(f"Configuration loaded: {config}")
        return config

    @property
    def tokenizer(self):
        raise NotImplementedError

    @property
    def model(self):
        raise NotImplementedError

    @staticmethod
    def push_config_to_hf(config: types.Config) -> None:
        api = HfApi()

        # TODO: This is still error since many object can't be serialized
        config_json = json.dumps(config.to_serializable_dict(), indent=2)

        api.upload_file(
            path_or_fileobj=config_json.encode(),
            path_in_repo="config.json",
            repo_id=config.training_kwargs.hub_model_id,
            repo_type="model",
        )
        logger.info(f"Config pushed to HuggingFace: {config.training_kwargs.hub_model_id}")

    @staticmethod
    def push_config_to_wandb(config: types.Config) -> None:
        wandb.init(
            project=os.getenv("PROJECT_NAME"),  # Replace with your actual project name
            name=config.training_kwargs.run_name,
            config=config.to_serializable_dict(),
        )
        logger.info(f"Config pushed to wandb: {config.training_kwargs.run_name}")

    def add_new_eos_token(self, eos_token: str, push_to_hub: bool = False):
        self._tokenizer.eos_token_id = [
            self._tokenizer.eos_token_id,
            self._tokenizer.convert_tokens_to_ids(eos_token),
        ]
        self._tokenizer.save_pretrained(self.config.training_kwargs.output_dir)
        if push_to_hub:
            self._tokenizer.push_to_hub(
                self.config.training_kwargs.hub_model_id,
                token=self.config.training_kwargs.hub_token,
                private=True,
            )

        logger.info(f"Add new eos token of {eos_token} completed")

    def create_repo(self, hub_model_id: str | None = None) -> None:
        hub_model_id = hub_model_id or self.config.training_kwargs.hub_model_id

        create_repo(
            hub_model_id,
            private=True,
        )
        logger.debug(f"Repo created: {self.config.training_kwargs.hub_model_id}")

    def memory_stats(self):
        memory_stats()
        logger.debug("Memory stats completed")
