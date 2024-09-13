import torch
from bitsandbytes.optim import AdamW8bit
from einops import rearrange
from peft import LoraConfig, get_peft_model
from supertrainer import logger, types
from supertrainer.evaluations.classification import compute_metrics
from supertrainer.trainers.base_trainer import BaseTrainer
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import wandb


class BERTTrainer(BaseTrainer):
    def __init__(self, config: types.Config, dataset: types.Dataset) -> None:
        config = self.postprocess_config(config)
        super().__init__(config, dataset)

    def postprocess_config(self, config):
        # Change design of this since it's weird to keep call super in here  even though it's guarantee that we will always call the super
        config = super().postprocess_config(config)

        classes = config.classes

        # create mapping and num of class first
        num_classes = len(classes)
        class2id = {class_: i for i, class_ in enumerate(classes)}
        id2class = {i: class_ for i, class_ in enumerate(classes)}

        config.num_classes = num_classes
        config.class2id = class2id
        config.id2class = id2class

        config.bitsandbytes_kwargs.bnb_4bit_compute_dtype = (
            config.bitsandbytes_kwargs.bnb_4bit_compute_dtype or "bfloat16"
            if torch.cuda.is_bf16_supported()
            else "float16"
        )
        quantization_config = BitsAndBytesConfig(**config.bitsandbytes_kwargs)

        config.model_kwargs.device_map = config.model_kwargs.device_map or {
            "": torch.cuda.current_device() if torch.cuda.is_available() else None
        }
        config.training_kwargs.torch_dtype = torch.bfloat16
        config.model_kwargs.attn_implementation = config.model_kwargs.attn_implementation or "sdpa"
        config.model_kwargs.quantization_config = quantization_config

        # amount of label
        config.model_kwargs.num_labels = num_classes

        config.peft_config = LoraConfig(
            **config.peft_kwargs,
        )

        # # Add HF to config
        config.training_kwargs.run_name += "-hf"

        return config

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification

            logger.debug("Lazy loading model")
            lora_config = LoraConfig(
                **self.config.peft_kwargs,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name, **self.config.model_kwargs
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            self._model = model
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            logger.info("Lazy loading tokenizer")
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            if self._tokenizer.pad_token is None:
                self._tokenizer.add_special_tokens({"pad_token": self._tokenizer.eos_token})
            if self._tokenizer.model_max_length > 100_000:
                self._tokenizer.model_max_length = 2048
        return self._tokenizer

    def train(self):
        if not self.config.testing:
            self.create_repo()
        logger.debug("Starting training process")

        logger.debug("Preparing dataset")
        dataset = self.dataset
        logger.debug("Initializing Trainer")

        eval_dataset = dataset["validation"] if not self.config.testing else None
        self.config.training_kwargs.do_eval = not self.config.testing

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            # model_init_kwargs=self.config.model_kwargs,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            args=TrainingArguments(
                **self.config.training_kwargs,
            ),
            data_collator=data_collator,
        )
        self.memory_stats()
        logger.debug("Starting training")

        trainer_stats = trainer.train()
        logger.debug(f"Training completed. Stats: {trainer_stats}")

        if not self.config.testing:
            # push configs
            self.push_config_to_hf(self.config)
            self.push_config_to_wandb(self.config)
            self.model.push_to_hub(
                self.config.training_kwargs.output_dir,
                self.tokenizer,
                save_method="lora",
            )
            # Save and push the updated tokenizer
            self.tokenizer.save_pretrained(self.config.training_kwargs.output_dir)
            self.tokenizer.push_to_hub(
                self.config.training_kwargs.hub_model_id,
                tokenizer=self.tokenizer,  # Pass the tokenizer explicitly
                save_method="lora",
                token=self.config.training_kwargs.hub_token,
                private=True,
            )
        print(trainer_stats)


class MLLMTrainer(BaseTrainer):
    def __init__(self, config: types.Config, dataset: types.Dataset) -> None:
        config = self.postprocess_config(config)
        super().__init__(config, dataset)

    def postprocess_config(self, config):
        # Change design of this since it's weird to keep call super in here  even though it's guarantee that we will always call the super
        config = super().postprocess_config(config)

        config.bitsandbytes_kwargs.bnb_4bit_compute_dtype = (
            config.bitsandbytes_kwargs.bnb_4bit_compute_dtype or "bfloat16"
            if torch.cuda.is_bf16_supported()
            else "float16"
        )
        quantization_config = BitsAndBytesConfig(**config.bitsandbytes_kwargs)

        config.model_kwargs.device_map = config.model_kwargs.device_map or {
            "": torch.cuda.current_device() if torch.cuda.is_available() else None
        }
        config.model_kwargs.attn_implementation = config.model_kwargs.attn_implementation or "sdpa"
        config.model_kwargs.quantization_config = quantization_config

        config.peft_config = LoraConfig(
            **config.peft_kwargs,
        )

        del config.model_kwargs.torch_dtype

        # TODO: THIS IS STILL HARDCODED, WE NEED TO CREATE TRANSFER CONFIG THINGY
        config.image_col = "image"

        # # Add HF to config
        config.training_kwargs.run_name += "-hf"

        return config

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM

            logger.debug("Lazy loading model")
            lora_config = LoraConfig(
                **self.config.peft_kwargs,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                revision=self.config.revision,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                **self.config.model_kwargs,
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
                self.config.model_name, revision=self.config.revision
            )
        return self._tokenizer

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

    def train(self):
        if not self.config.testing:
            self.create_repo()

        logger.debug("Starting training process")

        logger.debug("Preparing dataset")
        dataset = self.dataset
        logger.debug("Initializing Trainer")

        set_trainable_config = self.config.set_trainable

        if set_trainable_config:
            trainable_params_state_dict = self.set_trainable(
                trainable_params_names=set_trainable_config.trainable_params_names,
                set_other_trainable=set_trainable_config.set_other_trainable,
            )

        # TODO: HAVEN'T SUPPORT EVAL DATASET YET
        eval_dataset = None
        # self.config.training_kwargs.do_eval = not self.config.testing
        self.config.training_kwargs.do_eval = False

        # TODO: Move somewhere, need a lot of refactoring tho then .-.
        dataloaders = {
            key: torch.utils.data.DataLoader(
                dataset[key],
                batch_size=self.config.batch_size,
                shuffle=True if key != "test" else False,
                collate_fn=self.collate_fn,
            )
            for key in dataset.keys()
        }

        self.model.text_model.train()
        self.model.text_model.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=self.config.training_kwargs.gradient_checkpointing_kwargs
        )

        total_steps = (
            len(dataloaders["train"])
            * self.config.training_kwargs.num_train_epochs
            // self.config.training_kwargs.gradient_accumulation_steps
        )

        # Finetuning LoRA params
        lora_params = []
        for name, module in self.model.named_modules():
            if "lora" in name:
                lora_params.extend([p for p in module.parameters() if p.requires_grad])

        optimizer = AdamW8bit(
            [{"params": lora_params}],
            lr=self.config.training_kwargs.learning_rate,
            betas=(self.config.training_kwargs.adam_beta1, self.config.training_kwargs.adam_beta2),
            eps=self.config.training_kwargs.adam_epsilon,
        )

        # # For finetuning all text model params
        # optimizer = AdamW8bit(
        #     [{"params": self.model.text_model.parameters()}],
        #     lr=self.config.training_kwargs.learning_rate,
        #     betas=(self.config.training_kwargs.adam_beta1, self.config.training_kwargs.adam_beta2),
        #     eps=self.config.training_kwargs.adam_epsilon,
        # )

        if self.config.training_kwargs.report_to == "wandb":
            wandb.init(
                name=self.config.training_kwargs.run_name,
                project="ai701",
                config={
                    "epochs": self.config.training_kwargs.num_train_epochs,
                    "batch_size": self.config.batch_size,
                    "grad_accum_steps": self.config.training_kwargs.gradient_accumulation_steps,
                    "lr": self.config.training_kwargs.learning_rate,
                },
            )

        i = 0

        lr_scaling = self.config.peft_kwargs.lora_alpha / (self.config.peft_kwargs.r**0.5)
        for epoch in range(self.config.training_kwargs.num_train_epochs):
            for batch in tqdm(
                dataloaders["train"],
                desc=f"Epoch : {epoch + 1} / {self.config.training_kwargs.num_train_epochs}",
            ):
                i += 1
                loss = self.compute_loss(batch)
                loss.backward()

                if i % self.config.training_kwargs.gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                    optimizer.step()

                lr = self.linear_learning_rate_scheduler(
                    initial_lr=self.config.training_kwargs.learning_rate,
                    current_step=i,
                    warmup_steps=self.config.training_kwargs.warmup_steps,
                    total_steps=total_steps,
                )

                for param_group in optimizer.param_groups:
                    if param_group["params"] == lora_params:
                        param_group["lr"] = lr * lr_scaling
                    else:
                        param_group["lr"] = lr

                if i % self.config.training_kwargs.logging_steps == 0:
                    logger.info(f"Epoch: {epoch}, Loss: {loss.item()}")
                    if self.config.training_kwargs.report_to == "wandb":
                        wandb.log({"loss": loss.item()})

                if (
                    self.config.training_kwargs.do_eval
                    and i % self.config.training_kwargs.eval_steps == 0
                    and self.config.training_kwargs.report_to == "wandb"
                ):
                    val_loss = 0
                    for val_batch in tqdm(dataloaders["validation"], desc="Validation"):
                        with torch.no_grad():
                            val_loss += self.compute_loss(val_batch).item()
                    val_loss /= len(dataloaders["validation"])

                if self.config.training_kwargs.report_to == "wandb":
                    wandb.log(
                        {"train/loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
                        | (
                            {"val/loss": val_loss}
                            if self.config.training_kwargs.do_eval
                            and i % self.config.training_kwargs.eval_steps == 0
                            else {}
                        )
                    )
        if self.config.training_kwargs.report_to == "wandb":
            wandb.finish()

        self.model.save_pretrained(self.config.training_kwargs.output_dir)

        self.tokenizer.save_pretrained(self.config.training_kwargs.output_dir)

    def collate_fn(self, batch):
        images = [sample[self.config.image_col] for sample in batch]
        images = torch.stack(self.model.vision_encoder.preprocess(images))
        # I think p1 and p2 here is hardcoded for SigLIP
        images = rearrange(images, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14)
        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [self.tokenizer.bos_token_id]
            labs = [-100] * (self.config.num_imgs_tokens + 1)

            for qa in sample["qa"]:
                q_t = self.tokenizer(
                    f"\n\nQuestion: {qa['question']}\nAnswer: {qa['answer']}",
                    add_special_tokens=False,
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = self.tokenizer(
                    f"{qa['answer']}{self.config.answer_eos}", add_special_tokens=False
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
            images.to(dtype=self.config.dtype),
            torch.stack([torch.tensor(token, dtype=torch.long) for token in tokens_acc]),
            torch.stack([torch.tensor(label, dtype=torch.long) for label in labels_acc]),
            torch.stack([torch.tensor(attn_mask, dtype=torch.bool) for attn_mask in attn_mask_acc]),
        )

    def compute_loss(self, batch):
        images, tokens, labels, attn_mask = batch

        images = images.to(self.config.device)
        tokens = tokens.to(self.config.device)
        labels = labels.to(self.config.device)
        attn_mask = attn_mask.to(self.config.device)

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
