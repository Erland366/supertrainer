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

from tqdm import tqdm

from supertrainer import logger, type_hinting
from supertrainer.evaluations.bert import BertEvaluation
from supertrainer.inferences.llama import LlamaOutlinesInference
from supertrainer.utils.helpers import get_model_name


class LlamaEvaluation(BertEvaluation):
    def __init__(self, config: type_hinting.Config, dataset: type_hinting.Dataset):
        super().__init__(config, dataset)
        self.inference = LlamaOutlinesInference(self.config)

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
                    "true_label": self.config.dataset.id2class.get(true_label, "Unknown"),
                    "predicted_label": predicted_label.classes,
                    # "reasoning": predicted_label.reasoning,
                }
            )
        metrics = self.compute_metrics(results)
        logger.info(f"Evaluation metrics: {metrics}")

        self.save_results(results, metrics)
        return metrics
