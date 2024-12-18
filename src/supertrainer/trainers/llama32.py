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

from transformers import TrainingArguments
from trl import SFTTrainer

from supertrainer import logger, type_hinting
from supertrainer.trainers.base import BaseTrainer
from supertrainer.utils.helpers import remove_config_eval


class Llama32Trainer(BaseTrainer):
    def __init__(self, config: type_hinting.Config, dataset: type_hinting.Dataset) -> None:
        config = self.postprocess_config(config)
        super().__init__(config, dataset)

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        with config.allow_modification():
            if config.trainer.get("without_lora", False):
                config.trainer.without_lora = False
        config = super().postprocess_config(config)

        with config.allow_modification():
            config.trainer.training_kwargs.run_name += "-llm"

            # This both argument is set up in peft_kwargs of Unsloth
            del config.trainer.training_kwargs.gradient_checkpointing
            del config.trainer.training_kwargs.gradient_checkpointing_kwargs
        return config

    def _load_model_and_tokenizer(self):
        from unsloth import FastLanguageModel, get_chat_template

        logger.info("Lazy loading both model and tokenizer")
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.trainer.model_name,
            max_seq_length=self.config.trainer.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        self._model = FastLanguageModel.get_peft_model(
            self._model, **self.config.trainer.peft_kwargs
        )
        self._tokenizer = get_chat_template(self._tokenizer, self.config.dataset.chat_template)

    @property
    def model(self) -> "AutoModelForCausalLM" | "FastLanguageModel" | None:  # noqa # type: ignore
        if self._model is None or self._tokenizer is None:
            self._load_model_and_tokenizer()
        return self._model

    @property
    def tokenizer(self) -> "AutoTokenizer":  # noqa # type: ignore
        if self._model is None or self._tokenizer is None:
            self._load_model_and_tokenizer()
        return self._tokenizer

    def train(self) -> None:
        if not self.config.is_testing:
            self.create_repo()
            self.instantiate_wandb()

        logger.debug("Starting training process")

        logger.debug("Preparing dataset")
        dataset = self.dataset

        logger.debug("Initializing Trainer")
        train_dataset = dataset["train"]

        eval_dataset = None
        if not self.config.is_testing and dataset.get("validation", None) is not None:
            eval_dataset = dataset["validation"]

        with self.config.allow_modification():
            self.config.trainer.training_kwargs.do_eval = not self.config.is_testing
        if eval_dataset is None:
            logger.debug("No validation dataset found, skipping evaluation")
            remove_config_eval(self.config)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.trainer.max_seq_length,
            # dataset_num_proc=2,
            # Packing still buggy for unsloth (incorrect masking)
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                **self.config.trainer.training_kwargs,
            ),
        )
        self.memory_stats()

        logger.debug("Starting Training")
        trainer_stats = trainer.train()

        logger.debug("Training completed")
        logger.debug(f"Training completed. Stats: {trainer_stats}")

        if not self.config.is_testing:
            # push configs
            self.push_config_to_hf(self.config)
            self.push_config_to_wandb(self.config)
            self.model.save_pretrained_merged(
                self.config.trainer.training_kwargs.output_dir,
                self.tokenizer,
                save_method="lora",
            )
            self.model.push_to_hub_merged(
                self.config.trainer.training_kwargs.hub_model_id,
                self.tokenizer,
                save_method="lora",
            )
        print(trainer_stats)


class Llama32KTOTrainer(Llama32Trainer):
    def train(self) -> None:
        from trl import KTOConfig, KTOTrainer

        if not self.config.is_testing:
            self.create_repo()
            self.instantiate_wandb()

        logger.debug("Starting training process")

        logger.debug("Preparing dataset")
        dataset = self.dataset

        logger.debug("Initializing Trainer")
        train_dataset = dataset["train"]

        try:
            from unsloth import PatchKTOTrainer

            PatchKTOTrainer()
        except ImportError:
            pass

        assert "prompt" in train_dataset.column_names, "Prompt is required for KTO training"
        assert "completion" in train_dataset.column_names, "Completion is required for KTO training"
        assert "label" in train_dataset.column_names, "Label is required for KTO training"

        eval_dataset = None
        if not self.config.is_testing and dataset.get("validation", None) is not None:
            eval_dataset = dataset["validation"]

        with self.config.allow_modification():
            self.config.trainer.training_kwargs.do_eval = not self.config.is_testing

            # Error for KTO, disable it
            self.config.trainer.training_kwargs.include_num_input_tokens_seen = False
        if eval_dataset is None:
            logger.debug("No validation dataset found, skipping evaluation")
            remove_config_eval(self.config)

        trainer = KTOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # dataset_num_proc=2,
            # Packing still buggy for unsloth (incorrect masking)
            args=KTOConfig(
                **self.config.trainer.training_kwargs,
            ),
        )
        self.memory_stats()

        logger.debug("Starting Training")
        trainer_stats = trainer.train()

        logger.debug("Training completed")
        logger.debug(f"Training completed. Stats: {trainer_stats}")

        if not self.config.is_testing:
            # push configs
            self.push_config_to_hf(self.config)
            self.push_config_to_wandb(self.config)
            self.model.save_pretrained_merged(
                self.config.trainer.training_kwargs.output_dir,
                self.tokenizer,
                save_method="lora",
            )
            self.model.push_to_hub_merged(
                self.config.trainer.training_kwargs.hub_model_id,
                self.tokenizer,
                save_method="lora",
            )
        print(trainer_stats)
