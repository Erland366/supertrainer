from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from preskripsi_training import logger, types


class ABCDataset(ABC):
    dataset_name_or_path: str
    _tokenizer: None | AutoTokenizer = None
    _dataset: None | Dataset | DatasetDict = None

    @abstractmethod
    def prepare_dataset(self) -> Any:
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> AutoTokenizer:
        pass

    @property
    @abstractmethod
    def dataset(self) -> Dataset | DatasetDict:
        pass

    @abstractmethod
    def formatting_prompt_func(self):
        pass


class BaseDataset(ABCDataset):
    def __init__(self, config: types.Config, is_testing: bool = False) -> None:
        self.config = config
        self.is_testing = is_testing

    def test_tokenization(self, dataset):
        # I don't know if this is the best way to test tokenization
        logger.debug(f"Testing tokenization for {self.config.dataset_kwargs.path} on 5 examples")
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
            self._dataset = load_dataset(path=self.config.dataset_kwargs.path)
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
                self.config.dataset_kwargs.tokenizer_name_or_path
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
