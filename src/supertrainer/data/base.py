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

import os
from abc import ABC, abstractmethod

from datasets import Dataset, DatasetDict
from huggingface_hub import whoami
from transformers import AutoTokenizer

from supertrainer import SUPERTRAINER_ROOT, logger, type_hinting
from supertrainer.utils.helpers import load_dataset_plus_plus


class ABCDataset(ABC):
    dataset_name_or_path: str
    _tokenizer: None | AutoTokenizer = None
    _dataset: None | Dataset | DatasetDict = None

    # TODO: Thinking more about this!
    # @abstractmethod
    # def prepare_dataset(self) -> Any:
    #     pass

    @property
    @abstractmethod
    def tokenizer(self) -> AutoTokenizer:
        pass

    @property
    @abstractmethod
    def dataset(self) -> Dataset | DatasetDict:
        pass

    # TODO: Since we will assume that the dataset is prepared, we can remove this
    # @abstractmethod
    # def formatting_prompt_func(self):
    #     pass


class BaseDataset(ABCDataset):
    def __init__(self, config: type_hinting.Config, is_testing: bool = False) -> None:
        self.config = config
        self.is_testing = is_testing

    def test_tokenization(self, dataset):
        # I don't know if this is the best way to test tokenization
        logger.debug(
            f"Testing tokenization for {self.config.dataset.dataset_kwargs.path} on 5 examples"
        )
        for i in range(5):
            text = dataset["train"][i]["text"]
            tokens = self.tokenizer.encode(text)
            detokenized_text = self.tokenizer.decode(tokens)
            # Amount of words vs amount of tokens
            amount_of_words = len(text.split())
            amount_of_tokens = len(tokens)
            logger.debug(f"Tokens/words: {amount_of_tokens}/{amount_of_words}")

            # See first 20 tokens and last 20 tokens
            logger.debug(f"First 20 tokens: {tokens[:20]}")
            logger.debug(f"Last 20 tokens: {tokens[-20:]}")

            # Special tokens
            special_tokens = [token for token in tokens if token in self.tokenizer.all_special_ids]
            logger.debug(f"Special tokens: {special_tokens}")

            # Tokenization consistency
            is_consistent = text == detokenized_text
            logger.debug(f"Tokenization consistency: {is_consistent}")

    @property
    def dataset(self) -> Dataset | DatasetDict:
        if self._dataset is None:
            self._dataset = load_dataset_plus_plus(**self.config.dataset.dataset_kwargs)

            if self.config.is_testing:
                logger.debug("Dataset enter testing mode, only using 10 examples")
                has_subsets = "subsets" in self.config.dataset.dataset_kwargs

                if has_subsets:
                    # Handle nested DatasetDict(DatasetDict(Dataset)) structure
                    testing_dataset = DatasetDict()
                    for subset_name, subset_dict in self._dataset.items():
                        testing_dataset[subset_name] = DatasetDict(
                            train=subset_dict["train"].shuffle(seed=42).select(range(10))
                        )
                    self._dataset = testing_dataset
                else:
                    # Handle regular Dataset or DatasetDict
                    if isinstance(self._dataset, Dataset):
                        if len(self._dataset) > 10:
                            self._dataset = self._dataset.shuffle(seed=42).select(range(10))
                    else:
                        if len(self._dataset["train"]) > 10:
                            self._dataset = DatasetDict(
                                dict(
                                    train=self._dataset["train"].shuffle(seed=42).select(range(10))
                                )
                            )
        return self._dataset

    @staticmethod
    def print_text_after_substring(text: str, substring: str):
        """Prints the text after the first occurrence of the substring."""
        index = text.find(substring)
        if index != -1:
            print(text[index + len(substring) :])
        else:
            print("Substring not found.")

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.dataset.tokenizer_name_or_path
            )

            if self.config.dataset.get("chat_template", None) is not None:
                from unsloth import get_chat_template

                self._tokenizer = get_chat_template(
                    self._tokenizer, self.config.dataset.chat_template
                )
        return self._tokenizer

    def split_dataset(self, dataset: DatasetDict | Dataset) -> DatasetDict:
        logger.debug("Splitting dataset into train, test, and validation sets")

        train_test_split = dataset["train"].train_test_split(test_size=0.2)
        test_valid_split = train_test_split["test"].train_test_split(test_size=0.5)

        if self.is_testing:
            logger.debug("Split dataset enter testing mode, only using 10 examples")
            split_dataset = DatasetDict(
                {
                    "train": train_test_split["train"].shuffle(seed=42).select(range(10)),
                    "test": test_valid_split["test"].shuffle(seed=42).select(range(5)),
                    "validation": Dataset.from_list(
                        test_valid_split["train"].shuffle(seed=42).select(range(5))
                    ),
                }
            )
        else:
            split_dataset = DatasetDict(
                {
                    "train": train_test_split["train"].shuffle(seed=42),
                    "test": test_valid_split["test"].shuffle(seed=42),
                    "validation": Dataset.from_list(test_valid_split["train"].shuffle(seed=42)),
                }
            )

        return split_dataset


class ABCDatasetFormatter(ABC):
    @property
    @abstractmethod
    def dataset(self) -> Dataset | DatasetDict:
        pass

    # TODO: Is this clean?
    def process_dataset(self):
        self.transform_dataset()
        self.save_formatted_dataset()

    @abstractmethod
    def transform_dataset(self):
        pass

    @abstractmethod
    def save_formatted_dataset(self):
        pass


class BaseDatasetFormatter(ABCDatasetFormatter):
    @property
    def dataset(self) -> Dataset | DatasetDict:
        if self._dataset is None:
            self._dataset = load_dataset_plus_plus(path=self.config.dataset.dataset_kwargs.path)
        return self._dataset

    def save_formatted_dataset(self):
        logger.debug("Saving formatted dataset to disk")
        self.dataset.save_to_disk(
            os.path.join(
                os.getenv(SUPERTRAINER_ROOT),
                "datasets",
                self.config.dataset.dataset_kwargs.dataset_name,
            )
        )
        # TODO: Thinking about pushing the dataset to hub
        if self.config.dataset.dataset_kwargs.get("dataset_hub_id", None):
            logger.debug("Pushing dataset to hub")
            hf_id = whoami()["name"]
            self.dataset.push_to_hub(
                os.path.join(hf_id, self.config.dataset.dataset_kwargs.dataset_name)
            )
