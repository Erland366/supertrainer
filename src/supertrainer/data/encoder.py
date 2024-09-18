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

from datasets import DatasetDict

from supertrainer import logger, types
from supertrainer.data.base import BaseDataset


class EncoderDataset(BaseDataset):
    def __init__(self, config: types.Config, is_testing: bool = True) -> None:
        super().__init__(config, is_testing)

    def formatting_prompt_func(
        self, examples: list[types.Conversation], is_test_dataset: bool = True
    ):
        assert "input" in examples, "Missing input key"
        assert "output" in examples, "Missing output key"

        if "instruction" in examples:
            logger.warning(
                "We found instruction key, but we are not using it since we are "
                "using BERT-based model"
            )

        if is_test_dataset:
            pass

        inpts = examples["input"]
        outps = examples["output"]
        ents = examples["entity"]
        texts = []
        for ent, inp in zip(ents, inpts):
            text = f"{inp}: {ent}"
            texts.append(text)
        # SHIT THIS TOOK SO LONG, APPARENTLY THE KEY MUST BE `labels` AND CANNOT ANYTHING ELSE
        return {"text": texts, "labels": outps}

    def format_dataset(self, dataset: types.Conversation) -> types.Conversation:
        processed_dataset = DatasetDict()

        for split_name, split_dataset in dataset.items():
            is_test_dataset = split_name == "test"

            processed_split = split_dataset.map(
                lambda examples: self.formatting_prompt_func(
                    examples, is_test_dataset=is_test_dataset
                ),
                batched=True,
            )

            processed_dataset[split_name] = processed_split

        return processed_dataset
