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

from typing import List

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from supertrainer import logger, type_hinting
from supertrainer.inferences.base import BaseInference


class LLMInference(BaseInference):
    def __init__(self, config: type_hinting.Config) -> None:
        super().__init__(config)

    def load_model(self) -> AutoModelForCausalLM:
        lora_config = LoraConfig(**self.config.inference.peft_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.inference.model_name, **self.config.inference.model_kwargs
        )
        model = get_peft_model(model, lora_config)
        model.to(self.device)
        model.eval()
        logger.debug("LLM model loaded and ready for inference.")
        return model

    def load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.config.inference.model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        if tokenizer.model_max_length > 100_000:
            tokenizer.model_max_length = 2048
        return tokenizer

    def preprocess(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def postprocess(self, outputs):
        generated_ids = outputs
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text

    def postprocess_batch(self, outputs: torch.Tensor) -> List[str]:
        generated_texts = [
            self.tokenizer.decode(generated_id, skip_special_tokens=True)
            for generated_id in outputs
        ]
        return generated_texts

    def predict(self, prompt: str, max_length: int = 50) -> str:
        logger.debug(f"Generating text for prompt: {prompt}")
        inputs = self.preprocess(prompt)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
            )
        prediction = self.postprocess(outputs)
        logger.debug(f"Generated text: {prediction}")
        return prediction

    def predict_batch(self, prompts: List[str], max_length: int = 50) -> List[str]:
        logger.debug(f"Generating texts for batch of {len(prompts)} prompts.")
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
            )
        predictions = self.postprocess_batch(outputs)
        logger.debug(f"Generated texts: {predictions}")
        return predictions
