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
from PIL import Image
from transformers import Idefics3ForConditionalGeneration, Idefics3Processor

from src.supertrainer.inferences.chameleon import ChameleonInference
from supertrainer import logger, type_hinting
from supertrainer.utils.helpers import load_model_with_adaptive_attention, torch_dtype


class Idefics3Inference(ChameleonInference):
    def __init__(self, config: type_hinting.Config) -> None:
        self.config = self.postprocess_config(config)
        super().__init__(config)

    def load_model(self) -> "AutoModelForCausalLM":  # type: ignore # noqa: F821
        if self._model is None:
            logger.debug("Lazy loading model")

            peft_config = PeftConfig.from_pretrained(self.config.inference.model_name)

            model = load_model_with_adaptive_attention(
                Idefics3ForConditionalGeneration.from_pretrained,
                peft_config.base_model_name_or_path,
                trust_remote_code=True,
                **self.config.inference.model_kwargs,
            )

            # patch for Idefics3
            self.patch_clip_for_lora(model)

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
            if self.config.inference.processor_kwargs is None:
                with self.config.allow_modification():
                    self.config.inference.processor_kwargs = {}
            self._processor = Idefics3Processor.from_pretrained(
                self.config.inference.model_name,
                trust_remote_code=True,
                **self.config.inference.processor_kwargs,
            )

        return self._processor

    def preprocess(self, image: Image) -> type_hinting.Tensor:
        if self.config.trainer.get("prompt_template", None) is None:
            with self.config.allow_modification():
                self.config.trainer.prompt_template = "Answer briefly."
        prompt = self.config.trainer.get("prompt_template", "Answer briefly.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {"type": "image"},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.model.device, dtype=torch_dtype
        )

        return inputs

    def predict(self, image: Image) -> str:
        inputs = self.preprocess(image=image)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.config.inference.inference_kwargs,
            )

        prediction = self.postprocess(outputs[0][inputs["input_ids"].shape[1] :]).strip()
        return prediction
