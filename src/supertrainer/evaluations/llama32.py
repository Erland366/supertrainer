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
import re
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from supertrainer import logger, type_hinting
from supertrainer.evaluations.base import BaseEvaluation
from supertrainer.inferences.llama32 import Llama32Inference
from supertrainer.utils.helpers import get_model_name


class Llama32Evaluation(BaseEvaluation):
    def __init__(self, config: type_hinting.Config, dataset: type_hinting.Dataset):
        self.config = self.postprocess_config(config)
        self.inference = Llama32Inference(self.config)
        self.dataset = dataset

        self.prefix_patterns = [
            r"^i see (a |an )?",
            r"^this (appears|looks|seems) to be (a |an )?",
            r"^the image shows (a |an )?",
            r"^this is (a |an )?",
            r"^i believe this is (a |an )?",
            r"^the object (appears|looks|seems) to be (a |an )?",
            r"^it'?s (a |an )?",
            r"^there is (a |an )?",
            r"^we can see (a |an )?",
        ]
        self.suffix_patterns = [
            r"\s+in (the|this) (image|picture|photo)$",
            r"\s+object$",
            r"\s+here$",
        ]

        self.prefix_regex = re.compile("|".join(self.prefix_patterns), re.IGNORECASE)
        self.suffix_regex = re.compile("|".join(self.suffix_patterns), re.IGNORECASE)

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        with config.allow_modification():
            config.evaluation.classes = config.dataset.classes

        config = super().postprocess_config(config)
        return config

    def clean_prediction(self, prediction: str) -> tuple[str, bool]:
        """
        Clean the model prediction to extract the actual label.
        Returns tuple of (cleaned_label, is_unknown_flag)
        """
        prediction = prediction.lower().strip()

        prediction = self.prefix_regex.sub("", prediction)
        prediction = self.suffix_regex.sub("", prediction)
        prediction = prediction.strip()

        for label in self.config.dataset.classes:
            if prediction == label.lower():
                return label, False

        found_labels = []
        for label in self.config.dataset.classes:
            label_lower = label.lower()
            if label_lower in prediction:
                found_labels.append((prediction.index(label_lower), label))

        if found_labels:
            found_labels.sort()  # Sort by position
            return found_labels[0][1], False

        return "Unknown", True

    def evaluate(self):
        logger.info(f"Starting {get_model_name(self.inference.model)} evaluation")
        results = []
        unknown_predictions = []
        for i, data in enumerate(
            tqdm(self.dataset, desc=f"Evaluating {get_model_name(self.inference.model)}")
        ):
            text = data[self.config.dataset.text_col]
            label = data[self.config.dataset.label_col]
            raw_prediction = self.inference.predict(text)

            cleaned_prediction, is_unknown = self.clean_prediction(raw_prediction)
            result = {
                "text": text,
                "true_label": label,
                "predicted_label": cleaned_prediction,
                "raw_prediction": raw_prediction,
                "is_unknown": is_unknown,
            }
            results.append(result)

            if is_unknown:
                unknown_predictions.append(result)
        metrics = self.compute_metrics(results)
        logger.info(f"Evaluation metrics: {metrics}")

        # Add unknown prediction statistics to metrics
        metrics["unknown_count"] = len(unknown_predictions)
        metrics["unknown_percentage"] = (len(unknown_predictions) / len(results)) * 100

        logger.info(f"Evaluation metrics: {metrics}")
        logger.info(
            f"Found {metrics['unknown_count']} unknown predictions "
            f"({metrics['unknown_percentage']:.2f}% of total)"
        )

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
