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

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from supertrainer import logger, type_hinting


class BaseEvaluation(ABC):
    def __init__(self, config: type_hinting.Config, dataset: type_hinting.Dataset) -> None:
        self.config = self.postprocess_config(config)
        self.dataset = dataset

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        # Implement any common post-processing of config here
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
