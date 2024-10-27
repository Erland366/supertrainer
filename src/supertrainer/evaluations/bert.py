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
import json
import os
from datetime import datetime
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from supertrainer import SUPERTRAINER_PUBLIC_ROOT, logger, types
from supertrainer.evaluations.base import BaseEvaluation
from supertrainer.inferences.bert import BertInference
from supertrainer.utils.helpers import get_model_name


class BertEvaluation(BaseEvaluation):
    def __init__(self, config: types.Config, dataset: types.Dataset):
        self.config = self.postprocess_config(config)
        self.inference = BertInference(self.config)
        self.dataset = dataset

    def postprocess_config(self, config: types.Config) -> types.Config:
        with config.allow_modification():
            config.inference = config.evaluation
        return config

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
                    "true_label": true_label,
                    "predicted_label": self.config.dataset.class2id.get(predicted_label, "Unknown"),
                }
            )
        metrics = self.compute_metrics(results)
        logger.info(f"Evaluation metrics: {metrics}")

        self.save_results(results, metrics)
        return metrics

    def compute_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        true_labels = [result["true_label"] for result in results]
        predicted_labels = [result["predicted_label"] for result in results]

        # Get the list of classes from the configuration
        classes = self.config.evaluation.classes

        num_classes = len(classes)
        if num_classes == 2:
            average_method = "binary"
            # Set the positive class label (assuming the second class is positive)
            pos_label = classes[1]
        else:
            average_method = "micro"
            pos_label = None  # Not needed for multiclass when using 'micro' average

        # Compute metrics using scikit-learn
        accuracy = accuracy_score(true_labels, predicted_labels)
        if average_method == "binary":
            precision = precision_score(true_labels, predicted_labels, pos_label=pos_label)
            recall = recall_score(true_labels, predicted_labels, pos_label=pos_label)
            f1 = f1_score(true_labels, predicted_labels, pos_label=pos_label)
        else:
            precision = precision_score(
                true_labels, predicted_labels, average=average_method, labels=classes
            )
            recall = recall_score(
                true_labels, predicted_labels, average=average_method, labels=classes
            )
            f1 = f1_score(true_labels, predicted_labels, average=average_method, labels=classes)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def save_results(self, results: list[dict[str, Any]], metrics: dict[str, Any]):
        dataset_path = self.config.dataset.dataset_kwargs.path
        dataset_name = dataset_path.split("/")[-1]

        subset_name = self.config.dataset.dataset_kwargs.get("split", "")
        if subset_name:
            dataset_name += f"_{subset_name}"

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{dataset_name}_{current_time}"

        public_root = os.environ[SUPERTRAINER_PUBLIC_ROOT]

        output_folder = os.path.join(public_root, folder_name)

        os.makedirs(output_folder, exist_ok=True)

        results_file = os.path.join(output_folder, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        metrics_file = os.path.join(output_folder, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Metrics saved to {metrics_file}")
