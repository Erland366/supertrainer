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

from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from supertrainer import logger
from supertrainer.evaluations.classification import compute_metrics
from supertrainer.trainers.bert_trainer import BERTTrainer
from supertrainer.utils.helpers import remove_config_eval


class AraBERTTrainer(BERTTrainer):
    def train(self):
        if not self.config.is_testing:
            self.create_repo()
        logger.debug("Starting training process")

        logger.debug("Preparing dataset")
        dataset = self.dataset
        logger.debug("Initializing Trainer")

        subsets = self.config.dataset.get("subsets", [None])

        for subset in subsets:
            if subset:
                with self.config.allow_modification():
                    self.config.trainer.training_kwargs.run_name += f"_{subset}"

            train_data = dataset[subset]["train"] if subset else dataset["train"]
            eval_data = None
            if not self.config.is_testing and dataset.get("validation", None) is not None:
                eval_data = dataset[subset]["validation"] if subset else dataset["validation"]

            with self.config.allow_modification():
                self.config.trainer.training_kwargs.do_eval = not self.config.is_testing
            if eval_data is None:
                remove_config_eval(self.config)

            trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_data,
                eval_dataset=eval_data,
                compute_metrics=compute_metrics,
                args=TrainingArguments(**self.config.trainer.training_kwargs),
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            )

            # Train and log
            logger.debug("Starting training")
            trainer_stats = trainer.train()
            logger.debug(f"Training completed. Stats: {trainer_stats}")

            self.push_config_to_hf(self.config)
            self.push_config_to_wandb(self.config)

            output_dir = self.config.trainer.training_kwargs.output_dir
            hub_model_id = self.config.trainer.training_kwargs.hub_model_id

            for artifact in [self.model, self.tokenizer]:
                artifact.save_pretrained(output_dir)
                artifact.push_to_hub(hub_model_id, private=True)

            print(trainer_stats)
