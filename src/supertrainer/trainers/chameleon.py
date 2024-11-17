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
# ruff: noqa: E402


import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    ChameleonForConditionalGeneration,
    ChameleonProcessor,
    Trainer,
    TrainingArguments,
)
from unsloth_zoo.patching_utils import patch_torch_compile

from supertrainer import logger, type_hinting
from supertrainer.trainers.base_trainer import BaseMLLMTrainer
from supertrainer.utils.helpers import (
    import_class,
    load_model_with_adaptive_attention,
    remove_config_eval,
)


class ChameleonTrainer(BaseMLLMTrainer):
    def __init__(self, config: type_hinting.Config, dataset: type_hinting.Dataset) -> None:
        config = self.postprocess_config(config)
        super().__init__(config, dataset)

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        config = super().postprocess_config(config)

        with config.allow_modification():
            config.trainer.peft_config = LoraConfig(**config.trainer.peft_kwargs)
            config.trainer.training_kwargs.run_name += "-early_fusion"

        if config.dataset.get("image_col", None) is None:
            raise ValueError("Please provide the image column name in the dataset")

        if config.dataset.get("label_col", None) is None:
            raise ValueError("Please provide the label column name in the dataset")

        return config

    @property
    def model(self):
        if self._model is None:
            logger.debug("Lazy loading model")
            lora_config = self.config.trainer.peft_config
            model = load_model_with_adaptive_attention(
                ChameleonForConditionalGeneration.from_pretrained,
                self.config.trainer.model_name,
                trust_remote_code=True,
                **self.config.trainer.model_kwargs,
            )

            # checkpointing here!
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

            # semes like where do you make sure things? PEFT is a weird library!
            model = prepare_model_for_kbit_training(model)

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            if self.config.trainer.compile:
                patch_torch_compile()
                model = torch.compile(model)
            self._model = model
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            logger.debug("Lazy loading processor")
            if self.config.trainer.processor_kwargs is None:
                with self.config.allow_modification():
                    self.config.trainer.processor_kwargs = {}
            self._processor = ChameleonProcessor.from_pretrained(
                self.config.trainer.model_name,
                trust_remote_code=True,
                **self.config.trainer.processor_kwargs,
            )
        return self._processor

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

        data_collator = import_class(self.config.dataset.data_collator_class_name)(
            processor=self.processor,
            config=self.config,
        )

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                **self.config.trainer.training_kwargs,
            ),
            data_collator=data_collator,
            eval_dataset=eval_dataset,
            train_dataset=dataset["train"],
        )

        self.memory_stats()
        logger.debug("Starting training")

        trainer_stats = trainer.train()
        logger.debug(f"Training completed. Stats: {trainer_stats}")

        # push configs
        self.push_config_to_hf(self.config)
        self.push_config_to_wandb(self.config)
        self.model.save_pretrained(self.config.trainer.training_kwargs.output_dir)
        self.model.push_to_hub(self.config.trainer.training_kwargs.hub_model_id, private=True)
        # Save and push the updated tokenizer
        self.processor.save_pretrained(self.config.trainer.training_kwargs.output_dir)
        self.processor.push_to_hub(
            self.config.trainer.training_kwargs.hub_model_id,
            private=True,
        )
        print(trainer_stats)
