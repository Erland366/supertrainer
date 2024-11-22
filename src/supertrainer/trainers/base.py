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

import json
import os
from abc import ABC, abstractmethod

import torch
from huggingface_hub import HfApi, create_repo
from transformers import BitsAndBytesConfig

import wandb
from supertrainer import logger, type_hinting
from supertrainer.utils import memory_stats
from supertrainer.utils.helpers import remove_config_eval


class ABCTrainer(ABC):
    _model: "AutoModelForCausalLM" | "FastLanguageModel" | None = None  # noqa # type:ignore
    _tokenizer: "AutoTokenizer" | None = None  # noqa # type: ignore
    config: type_hinting.Config
    dataset: type_hinting.Dataset  # noqa # type: ignore

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
    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        return config


class BaseTrainer(ABCTrainer):
    def __init__(self, config: type_hinting.Config, dataset: type_hinting.Dataset) -> None:
        self.dataset = dataset
        self.config = config

    def train(self):
        raise NotImplementedError

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        # TODO: Move it from here since it's not modular enough
        with config.allow_modification():
            config.trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config.trainer.training_kwargs.fp16 = not torch.cuda.is_bf16_supported()
            config.trainer.training_kwargs.bf16 = torch.cuda.is_bf16_supported()
            config.trainer.training_kwargs.hub_model_id = (
                config.trainer.training_kwargs.hub_model_id
                + config.run_time
            )


            # mainly for data type casting purpose
            config.trainer.dtype = (
                torch.bfloat16 if config.trainer.training_kwargs.bf16 else torch.float36
            )

            # Construct the run_name
            model_name = config.trainer.model_name.split("/")[-1]
            dataset_name = config.dataset.dataset_kwargs.path.split("/")[-1]
            lora_rank = config.trainer.peft_kwargs.r
            learning_rate = config.trainer.training_kwargs.learning_rate
            num_epochs = config.trainer.training_kwargs.num_train_epochs

            run_name = (
                f"{model_name}_{dataset_name}_r{lora_rank}_lr{learning_rate}_e{num_epochs}_"
                f"{config.run_time}"
            )

            if config.is_testing:
                run_name = "TESTING_" + run_name

            if config.trainer.subset is not None:
                config.trainer.training_kwargs.hub_model_id += f"-{config.trainer.subset}"
                run_name += f"-{config.trainer.subset}"
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

            # FIX BUG WHERE MODEL_KWARGS IS NOT SET
            config.trainer.model_kwargs.device_map = config.trainer.model_kwargs.get(
                "device_map", None
            ) or {"": torch.cuda.current_device() if torch.cuda.is_available() else None}

            config.trainer.model_kwargs.quantization_config = quantization_config

            # if is testing
            if config.is_testing:
                remove_config_eval(config)

        logger.debug(f"Configuration loaded: {config}")
        return config

    def instantiate_wandb(self):
        wandb_kwargs = dict(tags=["testing"]) if self.config.is_testing else {}
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.trainer.training_kwargs.run_name,
            **wandb_kwargs,
        )

    @property
    def tokenizer(self):
        raise NotImplementedError

    @property
    def model(self):
        raise NotImplementedError

    @staticmethod
    def push_config_to_hf(config: type_hinting.Config) -> None:
        api = HfApi()

        # TODO: This is still error since many object can't be serialized
        config_json = json.dumps(config.to_serializable_dict(), indent=2)

        api.upload_file(
            path_or_fileobj=config_json.encode(),
            path_in_repo="config_hydra.json",
            repo_id=config.trainer.training_kwargs.hub_model_id,
            repo_type="model",
        )
        logger.info(f"Config pushed to HuggingFace: {config.trainer.training_kwargs.hub_model_id}")

    @staticmethod
    def push_config_to_wandb(config: type_hinting.Config) -> None:
        import json
        import tempfile

        logger.info(f"Config pushed to wandb: {config.trainer.training_kwargs.run_name}")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(config.to_serializable_dict(), f, indent=2)
            wandb.save(f.name)

    def add_new_eos_token(self, eos_token: str, push_to_hub: bool = False):
        self._tokenizer.eos_token_id = [
            self._tokenizer.eos_token_id,
            self._tokenizer.convert_tokens_to_ids(eos_token),
        ]
        self._tokenizer.save_pretrained(self.config.trainer.training_kwargs.output_dir)
        if push_to_hub:
            self._tokenizer.push_to_hub(
                self.config.trainer.training_kwargs.hub_model_id,
                token=self.config.trainer.training_kwargs.hub_token,
                private=True,
            )

        logger.info(f"Add new eos token of {eos_token} completed")

    def create_repo(self, hub_model_id: str | None = None) -> None:
        hub_model_id = hub_model_id or self.config.trainer.training_kwargs.hub_model_id

        create_repo(
            hub_model_id,
            private=True,
        )
        logger.debug(f"Repo created: {self.config.trainer.training_kwargs.hub_model_id}")

    def memory_stats(self):
        memory_stats()
        logger.debug("Memory stats completed")

    def set_trainable(
        self, trainable_params_names: list[str, str], set_other_trainable: bool | None = None
    ):
        for n, p in self.model.named_parameters():
            if any([name in n for name in trainable_params_names]):
                p.requires_grad_(True)
            else:
                if set_other_trainable is not None:
                    p.requires_grad_(set_other_trainable)

        trainable_params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        logger.debug(f"Trainable parameters: {trainable_params}")

        trainable_params_state_dict = {n: p.data for n, p in trainable_params.items()}

        return trainable_params_state_dict

    def reset(self):
        import gc

        import torch

        # Clear model
        if hasattr(self, "_model") and self._model is not None:
            logger.debug(f"Resetting model: {self._model.__class__.__name__}")
            self._model.cpu()
            del self._model
            self._model = None

        # Clear tokenizer
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            logger.debug(f"Resetting tokenizer: {self._tokenizer.__class__.__name__}")
            del self._tokenizer
            self._tokenizer = None

        if hasattr(self, "_processor") and self._processor is not None:
            logger.debug(f"Resetting processor: {self._processor.__class__.__name__}")
            del self._processor
            self._processor = None

        # Clear CUDA cache
        for _ in range(10):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Run garbage collector
            gc.collect()


class BaseMLLMTrainer(BaseTrainer):
    _processor: "AutoProcessor" | None = None  # noqa # type: ignore

    @property
    def processor(self):
        raise NotImplementedError

    def get_data_collator(self):
        raise NotImplementedError
