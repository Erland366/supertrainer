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
from bitsandbytes.optim import AdamW8bit
from einops import rearrange
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)

import wandb
from supertrainer import logger, type_hinting
from supertrainer.trainers.base_trainer import BaseTrainer


class MLLMTrainer(BaseTrainer):
    def __init__(self, config: type_hinting.Config, dataset: type_hinting.Dataset) -> None:
        config = self.postprocess_config(config)
        super().__init__(config, dataset)

    def postprocess_config(self, config):
        # Change design of this since it's weird to keep call super in here
        # even though it's guarantee that we will always call the super
        config = super().postprocess_config(config)

        # Set up lora config since we didn't use Unsloth
        with config.allow_modification():
            config.trainer.peft_config = LoraConfig(
                **config.trainer.peft_kwargs,
            )

            # Ugh this is disgusting
            del config.trainer.model_kwargs.torch_dtype

            config.dataset.image_col = "image"

            config.trainer.training_kwargs.run_name += "-mllm"

        return config

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM

            logger.debug("Lazy loading model")
            lora_config = LoraConfig(
                **self.config.trainer.peft_kwargs,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.trainer.model_name,
                revision=self.config.trainer.revision,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                **self.config.trainer.model_kwargs,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            self._model = model
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            logger.info("Lazy loading tokenizer")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.trainer.model_name, revision=self.config.trainer.revision
            )
        return self._tokenizer


    def train(self):
        if not self.config.is_testing:
            self.create_repo()

        logger.debug("Starting training process")

        logger.debug("Preparing dataset")
        dataset = self.dataset
        logger.debug("Initializing Trainer")

        set_trainable_config = self.config.trainer.set_trainable

        if set_trainable_config:
            self.set_trainable(  # type: ignore
                trainable_params_names=set_trainable_config.trainable_params_names,
                set_other_trainable=set_trainable_config.set_other_trainable,
            )

        # TODO: HAVEN'T SUPPORT EVAL DATASET YET

        with self.config.allow_modification():
            # self.config.training_kwargs.do_eval = not self.config.is_testing
            self.config.trainer.training_kwargs.do_eval = False

        # TODO: Move somewhere, need a lot of refactoring tho then .-.
        dataloaders = {
            key: torch.utils.data.DataLoader(
                dataset[key],
                batch_size=self.config.trainer.batch_size,
                shuffle=True if key != "test" else False,
                collate_fn=self.collate_fn,
            )
            for key in dataset.keys()
        }

        self.model.text_model.train()
        self.model.text_model.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=self.config.trainer.training_kwargs.gradient_checkpointing_kwargs
        )

        total_steps = (
            len(dataloaders["train"])
            * self.config.trainer.training_kwargs.num_train_epochs
            // self.config.trainer.training_kwargs.gradient_accumulation_steps
        )

        # Finetuning LoRA params
        lora_params = []
        for name, module in self.model.named_modules():
            if "lora" in name:
                lora_params.extend([p for p in module.parameters() if p.requires_grad])

        optimizer = AdamW8bit(
            [{"params": lora_params}],
            lr=self.config.trainer.training_kwargs.learning_rate,
            betas=(
                self.config.trainer.training_kwargs.adam_beta1,
                self.config.trainer.training_kwargs.adam_beta2,
            ),
            eps=self.config.trainer.training_kwargs.adam_epsilon,
        )

        # # For finetuning all text model params
        # optimizer = AdamW8bit(
        #     [{"params": self.model.text_model.parameters()}],
        #     lr=self.config.trainer.training_kwargs.learning_rate,
        #     betas=(self.config.trainer.training_kwargs.adam_beta1,
        #               self.config.trainer.training_kwargs.adam_beta2),
        #     eps=self.config.trainer.training_kwargs.adam_epsilon,
        # )

        if self.config.trainer.training_kwargs.report_to == "wandb" and not self.config.is_testing:
            # TODO: Fix this
            wandb.init(
                name=self.config.trainer.training_kwargs.run_name,
                project="ai701",
                config=self.config.to_serializable_dict(),
            )

        i = 0

        lr_scaling = self.config.trainer.peft_kwargs.lora_alpha / (
            self.config.trainer.peft_kwargs.r**0.5
        )
        for epoch in range(self.config.trainer.training_kwargs.num_train_epochs):
            for batch in tqdm(
                dataloaders["train"],
                desc=f"Epoch : {epoch + 1} / "
                f"{self.config.trainer.training_kwargs.num_train_epochs}",
            ):
                i += 1
                loss = self.compute_loss(batch)
                loss.backward()

                if i % self.config.trainer.training_kwargs.gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                    optimizer.step()

                lr = self.linear_learning_rate_scheduler(
                    initial_lr=self.config.trainer.training_kwargs.learning_rate,
                    current_step=i,
                    warmup_steps=self.config.trainer.training_kwargs.warmup_steps,
                    total_steps=total_steps,
                )

                for param_group in optimizer.param_groups:
                    if param_group["params"] == lora_params:
                        param_group["lr"] = lr * lr_scaling
                    else:
                        param_group["lr"] = lr

                if i % self.config.trainer.training_kwargs.logging_steps == 0:
                    logger.info(f"Epoch: {epoch}, Loss: {loss.item()}")
                    if self.config.trainer.training_kwargs.report_to == "wandb":
                        wandb.log({"loss": loss.item()})

                if (
                    self.config.trainer.training_kwargs.do_eval
                    and i % self.config.trainer.training_kwargs.eval_steps == 0
                    and self.config.trainer.training_kwargs.report_to == "wandb"
                ):
                    val_loss = 0
                    for val_batch in tqdm(dataloaders["validation"], desc="Validation"):
                        with torch.no_grad():
                            val_loss += self.compute_loss(val_batch).item()
                    val_loss /= len(dataloaders["validation"])

                if self.config.trainer.training_kwargs.report_to == "wandb":
                    wandb.log(
                        {"train/loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
                        | (
                            {"val/loss": val_loss}
                            if self.config.trainer.training_kwargs.do_eval
                            and i % self.config.trainer.training_kwargs.eval_steps == 0
                            else {}
                        )
                    )
        if self.config.trainer.training_kwargs.report_to == "wandb":
            wandb.finish()

        self.model.save_pretrained(self.config.trainer.training_kwargs.output_dir)

        self.tokenizer.save_pretrained(self.config.trainer.training_kwargs.output_dir)

    def collate_fn(self, batch):
        images = [sample[self.config.dataset.image_col] for sample in batch]
        images = torch.stack(self.model.vision_encoder.preprocess(images))
        # I think p1 and p2 here is hardcoded for SigLIP
        images = rearrange(images, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14)
        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [self.tokenizer.bos_token_id]
            labs = [-100] * (self.config.trainer.num_imgs_tokens + 1)

            for qa in sample["qa"]:
                q_t = self.tokenizer(
                    f"\n\nQuestion: {qa['question']}\nAnswer: {qa['answer']}",
                    add_special_tokens=False,
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = self.tokenizer(
                    f"{qa['answer']}{self.config.trainer.answer_eos}", add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)
            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))

        attn_mask_acc = []
        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([self.tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            images.to(dtype=self.config.trainer.dtype),
            torch.stack([torch.tensor(token, dtype=torch.long) for token in tokens_acc]),
            torch.stack([torch.tensor(label, dtype=torch.long) for label in labels_acc]),
            torch.stack([torch.tensor(attn_mask, dtype=torch.bool) for attn_mask in attn_mask_acc]),
        )

    def compute_loss(self, batch):
        images, tokens, labels, attn_mask = batch

        images = images.to(self.config.trainer.device)
        tokens = tokens.to(self.config.trainer.device)
        labels = labels.to(self.config.trainer.device)
        attn_mask = attn_mask.to(self.config.trainer.device)

        with torch.no_grad():
            img_embs = self.model.vision_encoder.encoder(images)
            img_embs = self.model.vision_encoder.projection(img_embs)

        tok_embs = self.model.text_model.get_input_embeddings()(tokens)
        # tok token at first I think is the start token
        input_embs = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

        outputs = self.model.text_model(
            inputs_embeds=input_embs,
            labels=labels,
            attention_mask=attn_mask,
        )

        return outputs.loss

    @staticmethod
    def linear_learning_rate_scheduler(
        initial_lr: float, current_step: int, warmup_steps: int, total_steps: int
    ) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return initial_lr * (current_step / warmup_steps)
        else:
            # Linear decay
            return initial_lr * max(
                0.0, (total_steps - current_step) / (total_steps - warmup_steps)
            )
