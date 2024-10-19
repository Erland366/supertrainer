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
from supertrainer.data.base import BaseDataset, BaseDatasetFormatter


class FactCheckingBERTEvaluationDataset(BaseDataset):
    def __init__(self, config: types.Config, is_testing: bool = False) -> None:
        super().__init__(self.postprocess_config(config), is_testing)
        self._is_prepared = None

    def postprocess_config(self, config: types.Config) -> types.Config:
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
