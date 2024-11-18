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

from __future__ import annotations

import torch
from PIL import Image

from supertrainer import logger, type_hinting
from supertrainer.inferences.base import BaseInferenceMLLM
from supertrainer.utils.helpers import load_model_with_adaptive_attention, torch_dtype


class ChameleonInference(BaseInferenceMLLM):
    def __init__(self, config: type_hinting.Config) -> None:
        self.config = self.postprocess_config(config)
        super().__init__(config)

        # Unsloth load model and tokenizer at the same time! Need buffer for them
        self.chat_template = "<image>Answer briefly.\n"

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        return config

    def load_model(self) -> "ChameleonForConditionalGeneration":  # type: ignore # noqa: F821
        if self._model is None:
            from peft import PeftConfig, PeftModel
            from transformers import ChameleonForConditionalGeneration

            peft_config = PeftConfig.from_pretrained(self.config.inference.model_name)

            model = load_model_with_adaptive_attention(
                ChameleonForConditionalGeneration.from_pretrained,
                peft_config.base_model_name_or_path,
                trust_remote_code=True,
                **self.config.inference.model_kwargs,
            )

            if not self.config.inference.base_only:
                model = PeftModel.from_pretrained(
                    model,
                    self.config.inference.model_name,
                    device_map=self.config.inference.model_kwargs.get("device_map", "auto"),
                )

            model.eval()
            self._model = model
        return self._model

    def load_processor(self) -> "AutoProcessor":  # type: ignore # noqa: F821
        if self._processor is None:
            from transformers import ChameleonProcessor

            if self.config.inference.processor_kwargs is None:
                with self.config.allow_modification():
                    self.config.inference.processor_kwargs = {}
            self._processor = ChameleonProcessor.from_pretrained(
                self.config.inference.model_name,
                trust_remote_code=True,
                **self.config.inference.processor_kwargs,
            )

        return self._processor

    def preprocess(self, image: Image) -> type_hinting.Tensor:
        prompt = f"<image>{self.config.evaluation.prompt_template}\n"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.model.device, dtype=torch_dtype
        )
        return inputs

    def postprocess(self, outputs: type_hinting.Tensor) -> str:
        # print("Generated tokens:")
        # for token_id in outputs:
        #     token = self.processor.decode([token_id])
        #     print(f"Token ID {token_id}: '{token}'")
        return self.processor.decode(outputs, skip_special_tokens=True)

    def predict(self, image: Image) -> str:
        inputs = self.preprocess(image=image)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.config.inference.inference_kwargs,
                forced_eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        prediction = self.postprocess(outputs[0][inputs["input_ids"].shape[1] :])
        return prediction

    def iterative_predict(self):
        """Run iterative inference in a loop."""
        logger.info("Starting iterative inference. Type 'exit' or 'quit' to stop.")
        from supertrainer.utils.helpers import load_dataset_plus_plus

        dataset = load_dataset_plus_plus(self.config.dataset.dataset_kwargs.path)
        dataset = dataset["test"]
        len_dataset = len(dataset)
        logger.info(f"Loaded dataset: {dataset}")

        try:
            while True:
                text = input("Enter idx for prediction: ").strip()
                if text.lower() in {"exit", "quit"}:
                    logger.info("Stopping iterative inference.")
                    break
                if not text:
                    print("Empty input. Please enter valid text.")
                    continue
                text = int(text)
                if text >= len_dataset and text < 0:
                    logger.info("Index out of range. Try input between 0 and {len_dataset}")
                    continue
                prediction = self.predict(dataset[text]["resized_image_64"])

                mapping = {
                    "ceramic": "hard",
                    "fabric": "soft",
                    "paper": "soft",
                    "glass": "soft",
                    "food": "soft",
                    "other": "hard",
                    "unknown": "hard",
                    "metal": "hard",
                    "wood": "hard",
                    "plastic": "hard",
                }
                print(f"Prediction: {mapping.get(prediction.lower(), 'hard')}")
        except KeyboardInterrupt:
            logger.info("Iterative inference interrupted by user.")
