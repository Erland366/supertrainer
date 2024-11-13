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
from peft import PeftConfig, PeftModel
from PIL import Image
from transformers import InstructBlipForConditionalGeneration

from supertrainer import type_hinting
from supertrainer.inferences.chameleon import ChameleonInference
from supertrainer.utils.helpers import load_model_with_adaptive_attention, torch_dtype


class InstructBlipInference(ChameleonInference):
    def __init__(self, config: type_hinting.Config) -> None:
        self.config = self.postprocess_config(config)
        super().__init__(config)

        # Instruct Blip can be used without chat template!
        self.chat_template = "<image>Answer briefly.\n"

    def load_model(self) -> "InstructBlipForConditionalGeneration":
        if self._model is None:
            peft_config = PeftConfig.from_pretrained(self.config.inference.model_name)

            # Error for Blip!
            with self.config.allow_modification():
                if self.config.inference.model_kwargs.get("use_cache", None) is not None:
                    del self.config.inference.model_kwargs.use_cache

            model = load_model_with_adaptive_attention(
                InstructBlipForConditionalGeneration.from_pretrained,
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

    def load_processor(self) -> "InstructionBlipProcessor":  # type: ignore # noqa: F821
        if self._processor is None:
            from transformers import InstructBlipProcessor

            if self.config.inference.processor_kwargs is None:
                with self.config.allow_modification():
                    self.config.inference.processor_kwargs = {}
            self._processor = InstructBlipProcessor.from_pretrained(
                self.config.inference.model_name,
                trust_remote_code=True,
                **self.config.inference.processor_kwargs,
            )
        return self._processor

    def postprocess(self, outputs: torch.Tensor) -> str:
        return self.processor.decode(outputs, skip_special_tokens=True)

    def preprocess(self, image: Image) -> type_hinting.Tensor:
        prompt = (
            "Question: What material is this object made of? "
            "Respond unknown if you are not sure. Short answer:"
        )

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.model.device, dtype=torch_dtype
        )
        return inputs

    def predict(self, image: Image) -> str:
        inputs = self.preprocess(image=image)

        with torch.no_grad():
            outputs = self.model.generate(
                # for instruct blip, we only pass the pixel values!
                **inputs,
                **self.config.inference.inference_kwargs,
                forced_eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        prediction = self.postprocess(outputs[0])
        return prediction
