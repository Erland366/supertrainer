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

from typing import Any

import torch

from supertrainer import logger, type_hinting
from supertrainer.data.base import BaseDataset


class InstructBlipDataCollator:
    def __init__(self, config: type_hinting.Config, processor: "AutoProcessor") -> None:  # noqa # type: ignore
        self.processor = processor
        self.config = config

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [
            self.config.dataset.id2class[example[self.config.dataset.label_col]]
            for example in examples
        ]
        images = [example[self.config.dataset.image_col] for example in examples]

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        batch["labels"] = batch["input_ids"].clone()
        return batch


class InstructBlipDataCollator2:
    def __init__(self, config: type_hinting.Config, processor: "AutoProcessor") -> None:  # noqa # type: ignore
        self.processor = processor
        self.config = config

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = (
            "Question: What material is this object made of? "
            "Respond unknown if you are not sure. Short answer:"
        )

        texts = [prompt] * len(examples)
        images = [example[self.config.dataset.image_col] for example in examples]

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # Process labels
        labels = [
            self.config.dataset.id2class[example[self.config.dataset.label_col]]
            for example in examples
        ]
        tokenized_labels = self.processor(text=labels, return_tensors="pt", padding=True)

        batch["labels"] = tokenized_labels["input_ids"]
        return batch


class Phi35VisionDataCollator:
    def __init__(self, config: type_hinting.Config, processor: "AutoProcessor") -> None:  # noqa # type: ignore
        self.processor = processor
        self.config = config

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        IGNORE_INDEX = -100
        all_input_ids = []
        all_label_ids = []
        all_pixel_values = []
        all_image_sizes = []

        for example in examples:
            image = example[self.config.dataset.image_col]
            answer = self.config.dataset.id2class[example[self.config.dataset.label_col]]

            prompt_message = {
                "role": "user",
                "content": "<|image_1|>\nAnswer briefly.",
            }

            prompt = self.processor.tokenizer.apply_chat_template(
                [prompt_message], tokenize=False, add_generation_prompt=True
            )
            answer = f"{answer}<|end|>\n<|endoftext|>"

            # Process each example
            batch = self.processor(prompt, [image], return_tensors="pt")
            prompt_input_ids = batch["input_ids"]

            # Do not add bos token to answer
            answer_input_ids = self.processor.tokenizer(
                answer, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]

            input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])).unsqueeze(0),
                    answer_input_ids,
                ],
                dim=1,
            )

            # Prepare for pad_sequence
            all_input_ids.append(input_ids.squeeze(0).unsqueeze(1))
            all_label_ids.append(labels.squeeze(0).unsqueeze(1))
            all_pixel_values.append(batch["pixel_values"])
            all_image_sizes.append(batch["image_sizes"])

        # Pad and combine all sequences
        input_ids = torch._C._nn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)

        labels = torch._C._nn.pad_sequence(
            all_label_ids, batch_first=True, padding_value=IGNORE_INDEX
        ).squeeze(2)

        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        pixel_values = torch.cat(all_pixel_values, dim=0)
        image_sizes = torch.cat(all_image_sizes, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }


class Florence2DataCollator:
    def __init__(self, config: type_hinting.Config, processor: "AutoProcessor") -> None:  # noqa # type: ignore
        self.processor = processor
        self.config = config

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        answers = []
        for example in examples:
            image = example[self.config.dataset.image_col]
            answer = self.config.dataset.id2class[example[self.config.dataset.label_col]]
            text = "<DocVQA>Answer briefly."
            texts.append(text)
            images.append([image])
            answers.append(answer)

        inputs = self.processor(
            text=list(texts), images=list(images), return_tensors="pt", padding=True
        )

        # Process labels
        labels = self.processor.tokenizer(
            text=answers, return_tensors="pt", padding=True, return_token_type_ids=False
        ).input_ids

        # Prepare the final batch
        batch_dict = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "labels": labels,
        }

        return batch_dict


class ChameleonDataCollator:
    def __init__(self, config: type_hinting.Config, processor: "AutoProcessor") -> None:  # noqa # type: ignore
        self.processor = processor
        self.config = config

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        image_token_id = self.processor.tokenizer.added_tokens_encoder.get("<image>", None)
        eos_token = self.processor.tokenizer.eos_token
        for example in examples:
            image = example[self.config.dataset.image_col]
            answer = self.config.dataset.id2class[example[self.config.dataset.label_col]]
            text = "<image>Answer briefly.\n" + answer + eos_token
            texts.append(text.strip())
            images.append([image])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch


class Idefics3DataCollator:
    def __init__(self, config: type_hinting.Config, processor: "AutoProcessor") -> None:  # noqa # type: ignore
        self.processor = processor
        self.config = config

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
            self.processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        for example in examples:
            image = example[self.config.dataset.image_col]
            answer = self.config.dataset.id2class[example[self.config.dataset.label_col]]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch


class SoftRoboticsTrainingDataset(BaseDataset):
    def __init__(self, config: type_hinting.Config, is_testing: bool = False) -> None:
        super().__init__(self.postprocess_config(config), is_testing)
        self._is_prepared = None

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        classes = config.dataset.classes
        num_classes = len(classes)
        class2id = {class_: i for i, class_ in enumerate(classes)}
        id2class = {i: class_ for i, class_ in enumerate(classes)}
        with config.allow_modification():
            config.dataset.class2id = class2id
            config.dataset.id2class = id2class
            config.dataset.num_classes = num_classes

        return config

    def prepare_dataset(self) -> "DatasetDict":  # noqa # type: ignore
        logger.debug("Preparing dataset")
        dataset = self.dataset

        logger.debug(f"Dataset loaded: {dataset}")

        logger.debug("MLLM Model preprocessing is in data collator, return the dataset as is")

        return dataset
