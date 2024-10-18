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

from typing import Any

import evaluate

from supertrainer import logger, types
from supertrainer.evaluations.base import BaseEvaluation
from supertrainer.inferences.sonnet import SonnetInstructorInference
from supertrainer.utils.helpers import get_model_name


class SonnetEvaluation(BaseEvaluation):
    def __init__(self, config: types.Config, dataset: types.Dataset):
        self.config = self.postprocess_config(config)
        self.inference = SonnetInstructorInference(self.config)
        self.dataset = dataset

    def postprocess_config(self, config: types.Config) -> types.Config:
        with config.allow_modification():
            config.inference = config.evaluation
        return config

    def evaluate(self):
        logger.info(f"Starting {get_model_name(self.inference.model)} evaluation")
        results = []
        for data in self.dataset:
            text = data["text"]
            true_label = data["labels"]
            predicted_label = self.inference.predict(text)
            results.append(
                {
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": self.config.dataset.class2id.get(
                        predicted_label, "Unknown"
                    ),
                }
            )
        metrics = self.compute_metrics(results)
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def compute_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        true_labels = [result["true_label"] for result in results]
        predicted_labels = [result["predicted_label"] for result in results]

        num_classes = len(self.config.evaluation.classes)
        if num_classes == 2:
            average_method = None
        else:
            average_method = "micro"

        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")

        accuracy = accuracy_metric.compute(predictions=predicted_labels, references=true_labels)
        precision = precision_metric.compute(
            predictions=predicted_labels, references=true_labels, average=average_method
        )
        recall = recall_metric.compute(
            predictions=predicted_labels, references=true_labels, average=average_method
        )
        f1 = f1_metric.compute(
            predictions=predicted_labels, references=true_labels, average=average_method
        )

        return {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1_score": f1["f1"],
        }
