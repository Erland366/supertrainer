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

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from supertrainer import logger, type_hinting
from supertrainer.evaluations.base import BaseEvaluation
from supertrainer.inferences.bert import BertInference
from supertrainer.utils.deprecation import deprecated
from supertrainer.utils.helpers import get_model_name


@deprecated(
    "This class is deprecated. Use respected model class instead! (e.g. AraBERT Evaluation)",
    alternative="AraBERTEvaluation",
)
class BertEvaluation(BaseEvaluation):
    def __init__(self, config: type_hinting.Config, dataset: type_hinting.Dataset):
        self.config = self.postprocess_config(config)
        self.inference = BertInference(self.config)
        self.dataset = dataset


    def evaluate(self):
        logger.info(f"Starting {get_model_name(self.inference.model)} evaluation")
        results = []
        for data in tqdm(self.dataset, desc=f"Evaluating {get_model_name(self.inference.model)}"):
            text = data["text"]
            true_label = data["labels"]
            predicted_label = self.inference.predict(text)
            results.append(
                {
                    "text": text,
                    "true_label": self.config.inference.id2class.get(true_label, "Unknown"),
                    "predicted_label": predicted_label,
                }
            )
        metrics = self.compute_metrics(results)
        logger.info(f"Evaluation metrics: {metrics}")

        self.save_results(results, metrics)
        return metrics

    def compute_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        true_labels = [result["true_label"] for result in results]
        predicted_labels = [result["predicted_label"] for result in results]

        classes = self.config.evaluation.classes

        num_classes = len(classes)
        if num_classes == 2:
            average_method = "binary"
            pos_label = classes[1]
        else:
            average_method = "macro"
            pos_label = None

        accuracy = accuracy_score(true_labels, predicted_labels)
        if average_method == "binary":
            precision = precision_score(
                true_labels, predicted_labels, pos_label=pos_label, zero_division=0
            )
            recall = recall_score(
                true_labels, predicted_labels, pos_label=pos_label, zero_division=0
            )
            f1 = f1_score(true_labels, predicted_labels, pos_label=pos_label, zero_division=0)
        else:
            precision = precision_score(
                true_labels,
                predicted_labels,
                average=average_method,
                labels=classes,
                zero_division=0,
            )
            recall = recall_score(
                true_labels,
                predicted_labels,
                average=average_method,
                labels=classes,
                zero_division=0,
            )
            f1 = f1_score(
                true_labels,
                predicted_labels,
                average=average_method,
                labels=classes,
                zero_division=0,
            )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
