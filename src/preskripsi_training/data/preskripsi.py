from __future__ import annotations

from datasets import DatasetDict
from transformers import AutoTokenizer

from preskripsi_training import logger, types
from preskripsi_training.data.encoder import EncoderDataset
from preskripsi_training.data.llm import LLMDataset
from preskripsi_training.data.templates.preskripsi_template import (
    DEFAULT_INPUT_TEMPLATE,
    DEFAULT_INSTRUCTION_TEMPLATE,
)


class PreskripsiBERTDataset(EncoderDataset):
    def __init__(self, config: types.Config, is_testing: bool = False) -> None:
        super().__init__(config, is_testing)

    def format_for_aspect_sentiment_analysis(self, dataset: DatasetDict) -> DatasetDict:
        logger.debug("Formatting dataset for aspect sentiment analysis")

        def format_example(example: list[types.Conversation]):
            entity = example["Entity"]
            input_text = DEFAULT_INPUT_TEMPLATE.format(
                title=example["Title"], content=example["Content"]
            )
            output_text = self.config.class2id[example["Sentiment Correction"].lower()]
            return {
                "entity": entity,
                "input": input_text,
                "output": output_text,
            }

        formatted_dataset = dataset.map(format_example, batched=False)
        logger.debug("Removing unnecessary columns")
        formatted_dataset = DatasetDict(
            {
                split: ds.remove_columns(
                    [col for col in ds.column_names if col not in ["input", "output", "entity"]]
                )
                for split, ds in formatted_dataset.items()
            }
        )
        logger.debug("Dataset formatting for aspect sentiment analysis completed")
        return formatted_dataset

    def tokenized_dataset(self, dataset: DatasetDict) -> DatasetDict:
        def tokenize_map(examples: types.Conversation, tokenizer: "AutoTokenizer"):  # noqa # type: ignore
            # TODO: FIx this!
            tokenizer.truncation_side = "left"
            return tokenizer(examples["text"], truncation=True)

        tokenized_dataset = dataset.map(
            lambda examples: tokenize_map(examples, self.tokenizer), batched=True
        )
        return tokenized_dataset

    def prepare_dataset(self):
        logger.debug("Preparing dataset")
        dataset_dict = self.dataset
        logger.debug(f"Dataset loaded: {dataset_dict}")

        logger.debug("Splitting dataset")
        split_dataset = self.split_dataset(dataset_dict)
        logger.debug(f"Dataset split: {split_dataset}")

        logger.debug("Formatting dataset for aspect sentiment analysis")
        formatted_dataset = self.format_for_aspect_sentiment_analysis(split_dataset)
        logger.debug(f"Dataset formatted: {formatted_dataset}")

        logger.debug("Applying formatting prompts")
        dataset = self.format_dataset(formatted_dataset)
        self.test_tokenization(dataset)

        dataset = self.tokenized_dataset(dataset)

        # Print some examples from the dataset to inspect the tokenizer's output
        print("*** Example from the dataset ***")
        for i in range(5):
            print(f"Example {i+1}:")
            self.print_text_after_substring(dataset["train"][i]["text"], "[/Judul]")
            print("-" * 20)

        logger.debug("Dataset preparation completed")

        return dataset


class PreskripsiDataset(LLMDataset):
    def __init__(self, config: types.DictConfig, is_testing: bool = True) -> None:
        super().__init__(config, is_testing)

    @staticmethod
    def format_for_aspect_sentiment_analysis(dataset: DatasetDict) -> DatasetDict:
        logger.debug("Formatting dataset for aspect sentiment analysis")

        def format_example(example: list[types.Conversation]) -> dict[str, str]:
            instruction = DEFAULT_INSTRUCTION_TEMPLATE.format(entity=example["Entity"])
            input_text = DEFAULT_INPUT_TEMPLATE.format(
                title=example["Title"], content=example["Content"]
            )
            output_text = example["Sentiment Correction"]
            return {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }

        formatted_dataset = dataset.map(format_example, batched=False)
        logger.debug("Removing unnecessary columns")
        formatted_dataset = DatasetDict(
            {
                split: ds.remove_columns(
                    [
                        col
                        for col in ds.column_names
                        if col not in ["instruction", "input", "output", "label"]
                    ]
                )
                for split, ds in formatted_dataset.items()
            }
        )
        logger.debug("Dataset formatting for aspect sentiment analysis completed")
        return formatted_dataset

    def prepare_dataset(self):
        logger.debug("Preparing dataset")
        dataset_dict = self.dataset
        logger.debug(f"Dataset loaded: {dataset_dict}")

        logger.debug("Splitting dataset")
        split_dataset = self.split_dataset(dataset_dict)
        logger.debug(f"Dataset split: {split_dataset}")

        logger.debug("Formatting dataset for aspect sentiment analysis")
        formatted_dataset = self.format_for_aspect_sentiment_analysis(split_dataset)
        logger.debug(f"Dataset formatted: {formatted_dataset}")

        logger.debug("Applying formatting prompts")
        dataset = self.format_dataset(formatted_dataset)
        self.test_tokenization(dataset)

        # Print some examples from the dataset to inspect the tokenizer's output
        print("*** Example from the dataset ***")
        for i in range(5):
            print(f"Example {i+1}:")
            self.print_text_after_substring(dataset["train"][i]["text"], "[/Judul]")
            print("-" * 20)

        logger.debug("Dataset preparation completed")

        return dataset
