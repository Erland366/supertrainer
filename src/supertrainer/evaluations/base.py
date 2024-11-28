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

# src/supertrainer/evaluations/base_evaluation.py

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from supertrainer import SUPERTRAINER_PUBLIC_ROOT, logger, type_hinting


class BaseEvaluation(ABC):
    def __init__(self, config: type_hinting.Config, dataset: type_hinting.Dataset) -> None:
        self.config = self.postprocess_config(config)
        self.dataset = dataset

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        with config.allow_modification():
            if config.evaluation.subset is not None:
                config.evaluation.model_name = (
                    f"{config.evaluation.model_name}-{config.evaluation.subset}"
                )
                logger.info(f"Found subset! Model name updated to {config.evaluation.model_name}")

            config.inference = config.evaluation
        return config

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Perform evaluation and return metrics."""
        pass

    @abstractmethod
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute and return evaluation metrics based on results."""
        pass

    def run_evaluation(self):
        """Run the evaluation process."""
        logger.info("Starting evaluation process.")
        metrics = self.evaluate()
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics

    def save_results(self, results: list[dict[str, Any]], metrics: dict[str, Any]):
        dataset_path = self.config.dataset.dataset_kwargs.path
        dataset_name = dataset_path.split("/")[-1]

        subset_name = self.config.dataset.dataset_kwargs.get("split", "")
        if subset_name:
            dataset_name += f"-{subset_name}"

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{dataset_name}-{current_time}"

        model_name = (self.config.evaluation.model_name).split("/")[-1]
        folder_name += f"-{model_name}"

        if self.config.evaluation.subset is not None:
            folder_name += f"-{self.config.evaluation.subset}"

        if self.config.evaluation.get("base_only", None) is not None:
            if self.config.evaluation.base_only:
                folder_name += "-base_only"

        public_root = os.environ[SUPERTRAINER_PUBLIC_ROOT]

        output_folder = os.path.join(public_root, folder_name)

        os.makedirs(output_folder, exist_ok=True)

        results_file = os.path.join(output_folder, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        metrics_file = os.path.join(output_folder, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        config_file = os.path.join(output_folder, "config.json")
        with open(config_file, "w") as f:
            json.dump(self.config.to_serializable_dict(), f, indent=4)

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Metrics saved to {metrics_file}")
        logger.info(f"Config saved to {config_file}")
