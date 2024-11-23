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

from supertrainer import logger, type_hinting
from supertrainer.data.base import BaseDataset, BaseDatasetFormatter


class FactCheckingTrainingLLMDataset(BaseDataset):
    def __init__(self, config: type_hinting.Config, is_testing: bool = False) -> None:
        super().__init__(self.postprocess_config(config), is_testing)

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        assert (
            config.dataset.get("chat_template", None) is not None
        ), "chat_template is required in self.config.dataset.chat_template"

        # I don't think we want to convert the classes to id here
        # Maybe just make it always lowercase?

        return config

    def formatting_prompt_func(
        self,
        examples: list[type_hinting.Conversation],
        use_default_system_prompt: bool = True,
        is_test_dataset: bool = True,
    ) -> dict[str, str]:
        """Batch formatting right here"""

        conversations = []

        texts = examples[self.config.dataset.text_col]
        labels = examples[self.config.dataset.label_col]
        for text, label in zip(texts, labels):
            conversation = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": label},
            ]

            forbidden_system_prompts = ["llama-3.1", "llama-31"]

            if (
                use_default_system_prompt
                and "system" not in conversation[0].values()
                and self.config.dataset.chat_template not in forbidden_system_prompts
            ):
                conversation.insert(
                    0,
                    {
                        "role": "system",
                        "content": self.config.dataset.get(
                            "default_system_prompt", "You are an helpful AI assistant"
                        ),
                    },
                )

            conversations.append(conversation)

        texts = []
        for convos in conversations:
            if is_test_dataset:
                logger.warning_once(
                    "Remove assistant output from test dataset "
                    "instead, add the generation prompt"
                )
                convos = convos[:-1]
                text = self.tokenizer.apply_chat_template(
                    convos, tokenize=False, add_generation_prompt=True
                )
            else:
                text = self.tokenizer.apply_chat_template(convos, tokenize=False)
            texts.append(text)
        return {
            "text": texts,
        }

    def format_dataset(self, dataset: type_hinting.Conversation) -> type_hinting.Conversation:
        processed_dataset = DatasetDict()
        subsets = self.config.dataset.dataset_kwargs.get("subsets", [None])

        def process_split(split_dataset, split_name):
            is_test_dataset = split_name == "test"
            return split_dataset.map(
                lambda examples: self.formatting_prompt_func(
                    examples, is_test_dataset=is_test_dataset
                ),
                batched=True,
            )

        for subset in subsets:
            current_dataset = dataset[subset] if subset is not None else dataset

            if subset is not None:
                # Handle nested structure for named subsets
                processed_subset = DatasetDict()
                for split_name, split_dataset in current_dataset.items():
                    processed_subset[split_name] = process_split(split_dataset, split_name)
                processed_dataset[subset] = processed_subset
            else:
                # Handle flat structure for no subset
                for split_name, split_dataset in current_dataset.items():
                    processed_dataset[subset] = process_split(split_dataset, split_name)

        return processed_dataset

    def prepare_dataset(self) -> "DatasetDict":  # noqa # type: ignore
        logger.debug("Preparing dataset")
        dataset = self.dataset

        # BUG! Will fix this
        # self.test_tokenization(dataset)
        dataset = self.format_dataset(dataset)

        logger.debug(f"Dataset loaded: {dataset}")

        return dataset


class FactCheckingTrainingDataset(BaseDataset):
    def __init__(self, config: type_hinting.Config, is_testing: bool = False) -> None:
        super().__init__(self.postprocess_config(config), is_testing)

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

    def format_dataset(self, dataset: DatasetDict) -> DatasetDict:
        def tokenize_and_map(examples: dict, tokenizer: "AutoTokenizer"):  # noqa # type: ignore
            tokenizer.truncation_side = "left"
            tokenized = tokenizer(
                examples[self.config.dataset.text_col], truncation=True, padding=True
            )

            if self.config.dataset.label_col in examples:
                tokenized[self.config.dataset.label_col] = [
                    self.config.dataset.class2id[label]
                    for label in examples[self.config.dataset.label_col]
                ]

            return tokenized

        if self.config.dataset.dataset_kwargs.get("subsets", [None]) != [None]:
            # Handle nested DatasetDict
            processed_dataset = {}
            for subset_key, subset_dict in dataset.items():
                processed_subset = {}
                for split_key, split_dataset in subset_dict.items():
                    processed_subset[split_key] = split_dataset.map(
                        lambda x: tokenize_and_map(x, self.tokenizer),
                        batched=True,
                        remove_columns=split_dataset.column_names,
                    )
                processed_dataset[subset_key] = DatasetDict(processed_subset)
            return DatasetDict(processed_dataset)
        else:
            # Handle flat DatasetDict
            return dataset.map(
                lambda x: tokenize_and_map(x, self.tokenizer),
                batched=True,
                remove_columns=dataset["train"].column_names,
            )

    def prepare_dataset(self) -> "DatasetDict":  # noqa # type: ignore
        logger.debug("Preparing dataset")
        dataset = self.dataset

        # BUG! Will fix this
        # self.test_tokenization(dataset)
        dataset = self.format_dataset(dataset)

        logger.debug(f"Dataset loaded: {dataset}")

        return dataset


class FactCheckingBERTEvaluationDataset(BaseDataset):
    def __init__(self, config: type_hinting.Config, is_testing: bool = False) -> None:
        super().__init__(self.postprocess_config(config), is_testing)
        self._is_prepared = None

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        classes = config.evaluation.classes
        num_classes = len(classes)
        class2id = {class_: i for i, class_ in enumerate(classes)}
        id2class = {i: class_ for i, class_ in enumerate(classes)}
        with config.allow_modification():
            config.dataset.class2id = class2id
            config.dataset.id2class = id2class
            config.dataset.num_classes = num_classes

        return config

    def format_for_evaluation(self, dataset: DatasetDict) -> DatasetDict:
        logger.debug("Formatting dataset for evaluation")

        def format_example(example):
            text = example["text"]
            label = self.config.dataset.class2id[example["labels"]]
            return {"text": text, "labels": label}

        formatted_dataset = dataset.map(format_example)
        logger.debug("Removing unnecessary columns")
        formatted_dataset = formatted_dataset.remove_columns(
            [col for col in formatted_dataset.column_names if col not in ["text", "labels"]]
        )

        return formatted_dataset

    def prepare_dataset(self):
        logger.debug("Preparing dataset")
        dataset = self.dataset

        if isinstance(dataset, DatasetDict):
            logger.debug("Found a DatasetDict, we will use the test split")
            dataset = dataset["test"]

        logger.debug(f"Dataset loaded: {dataset}")

        logger.debug("Formatting dataset for fact checking")
        formatted_dataset = self.format_for_evaluation(dataset)

        return formatted_dataset


class FactCheckingBERTEvaluationDatasetFormatter(BaseDatasetFormatter):
    @staticmethod
    def format_dataset(examples):
        """
        Formats the dataset by combining claims and evidence, and extracting labels.
        """
        texts = [
            f"{claim}. Evidence: {evidence}"
            for claim, evidence in zip(examples["claim"], examples["evidence"])
        ]
        labels = examples["evidence_label"]
        return {"text": texts, "labels": labels}

    def transform_dataset(self):
        """
        Applies formatting to the loaded dataset.
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Please call load_dataset() first.")
        self.formatted_dataset = self.dataset.map(
            self.format_dataset, batched=True, remove_columns=self.dataset.column_names
        )
        return self.formatted_dataset


class FactCheckingSonnetEvaluationDataset(FactCheckingBERTEvaluationDataset):
    # This dataset is used for the Sonnet model
    # Should be the same here
    pass


class FactCheckingGPTEvaluationDataset(FactCheckingBERTEvaluationDataset):
    # This dataset is used for the Sonnet model
    # Should be the same here
    pass


class FactCheckingGemmaEvaluationDataset(FactCheckingBERTEvaluationDataset):
    # This dataset is used for the Gemma model
    # Should be the same here
    pass


class FactCheckingMistralEvaluationDataset(FactCheckingBERTEvaluationDataset):
    # This dataset is used for the Mistral model
    # Should be the same here
    pass


class FactCheckingLlamaEvaluationDataset(FactCheckingBERTEvaluationDataset):
    # This dataset is used for the Mistral model
    # Should be the same here
    pass


class FactCheckingQwenEvaluationDataset(FactCheckingBERTEvaluationDataset):
    # This dataset is used for the Mistral model
    # Should be the same here
    pass
