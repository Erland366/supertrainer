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

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from unsloth_zoo.patching_utils import patch_torch_compile

from supertrainer import logger, type_hinting
from supertrainer.evaluations.classification import compute_metrics
from supertrainer.trainers.base_trainer import BaseTrainer
from supertrainer.utils.helpers import load_model_with_adaptive_attention, remove_config_eval


class BERTTrainer(BaseTrainer):
    def __init__(self, config: type_hinting.Config, dataset: type_hinting.Dataset) -> None:
        config = self.postprocess_config(config)
        super().__init__(config, dataset)

    def postprocess_config(self, config):
        # Change design of this since it's weird to keep call super in here
        # even though it's guarantee that we will always call the super
        config = super().postprocess_config(config)

        classes = config.dataset.classes

        # create mapping and num of class first
        num_classes = len(classes)
        class2id = {class_: i for i, class_ in enumerate(classes)}
        id2class = {i: class_ for i, class_ in enumerate(classes)}

        with config.allow_modification():
            config.trainer.num_classes = num_classes
            config.trainer.class2id = class2id
            config.trainer.id2class = id2class

            # amount of label
            config.trainer.model_kwargs.num_labels = num_classes

            # Set up lora config since we didn't use Unsloth
            config.trainer.peft_config = LoraConfig(
                **config.trainer.peft_kwargs,
            )

            # Add HF to config
            config.trainer.training_kwargs.run_name += "-bert"

        return config

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification

            logger.debug("Lazy loading model")
            lora_config = LoraConfig(
                **self.config.trainer.peft_kwargs,
            )
            model = load_model_with_adaptive_attention(
                AutoModelForSequenceClassification,
                self.config.trainer.model_name,
                **self.config.trainer.model_kwargs,
            )

            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

            model = prepare_model_for_kbit_training(model)

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            if self.config.trainer.compile:
                patch_torch_compile()
                model = torch.compile(model)
            self._model = model
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            logger.info("Lazy loading tokenizer")
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.trainer.model_name)

            if self._tokenizer.pad_token is None:
                self._tokenizer.add_special_tokens({"pad_token": self._tokenizer.eos_token})
            if self._tokenizer.model_max_length > 100_000:
                self._tokenizer.model_max_length = 2048
        return self._tokenizer

    def train(self):
        if not self.config.is_testing:
            self.create_repo()
        logger.debug("Starting training process")

        logger.debug("Preparing dataset")
        dataset = self.dataset
        logger.debug("Initializing Trainer")

        if dataset.get("validation", None) is None or self.config.is_testing:
            eval_dataset = None
            remove_config_eval(self.config)
        else:
            eval_dataset = dataset["validation"]

        with self.config.allow_modification():
            self.config.trainer.training_kwargs.do_eval = not self.config.is_testing

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            # model_init_kwargs=self.config.model_kwargs,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            args=TrainingArguments(
                **self.config.trainer.training_kwargs,
            ),
            data_collator=data_collator,
        )
        self.memory_stats()
        logger.debug("Starting training")

        trainer_stats = trainer.train()
        logger.debug(f"Training completed. Stats: {trainer_stats}")

        self.push_config_to_hf(self.config)
        self.push_config_to_wandb(self.config)
        self.model.save_pretrained(self.config.trainer.training_kwargs.output_dir)
        self.model.push_to_hub(self.config.trainer.training_kwargs.hub_model_id, private=True)
        # Save and push the updated tokenizer
        self.tokenizer.save_pretrained(self.config.trainer.training_kwargs.output_dir)
        self.tokenizer.push_to_hub(
            self.config.trainer.training_kwargs.hub_model_id,
            private=True,
        )
        print(trainer_stats)
