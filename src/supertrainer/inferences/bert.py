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

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from supertrainer import logger, type_hinting
from supertrainer.inferences.base import BaseInference
from supertrainer.utils.deprecation import deprecated
from supertrainer.utils.helpers import load_model_with_adaptive_attention


@deprecated(
    "This class is deprecated. Use respected model class instead! (e.g. AraBERT Inference)",
    alternative="AraBERT Inference",
)
class BertInference(BaseInference):
    def __init__(self, config: type_hinting.Config) -> None:
        super().__init__(config)

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        id2class = {k: v for k, v in enumerate(config.inference.classes)}
        class2id = {v: k for k, v in id2class.items()}

        with config.allow_modification():
            config.inference.id2class = id2class
            config.inference.class2id = class2id

        return config

    def load_model(self) -> AutoModelForSequenceClassification:
        peft_config = PeftConfig.from_pretrained(self.config.inference.model_name)

        model = load_model_with_adaptive_attention(
            AutoModelForSequenceClassification.from_pretrained,
            peft_config.base_model_name_or_path,
            num_labels=len(self.config.inference.classes),
            trust_remote_code=True,
            **self.config.inference.model_kwargs,
        )

        if not self.config.inference.base_only:
            logger.info(f"Loading PEFT model: {self.config.inference.model_name}")
            model = PeftModel.from_pretrained(
                model,
                self.config.inference.model_name,
                device_map=self.config.inference.model_kwargs.get("device_map", "auto"),
            )

        model.eval()
        return model

    def load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.inference.model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        if tokenizer.model_max_length > 100_000:
            tokenizer.model_max_length = 2048
        return tokenizer

    def preprocess(self, text: str):
        return self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(self.device)

    def postprocess(self, outputs):
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        label = self.config.inference.id2class.get(predicted_class, "Unknown")
        return label

    def predict(self, text: str) -> str:
        inputs = self.preprocess(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction = self.postprocess(outputs)
        return prediction
