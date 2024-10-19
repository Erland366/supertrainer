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

from supertrainer import types
from supertrainer.inferences.base import BaseInference, BaseOutlinesInference


class LlamaInference(BaseInference):
    def __init__(self, config: types.Config) -> None:
        self.config = self.postprocess_config(config)
        super().__init__(config)

        # Unsloth load model and tokenizer at the same time! Need buffer for them
        self._buffer_model = None
        self._buffer_tokenizer = None
        self.chat_template = "llama-3.1"

    def postprocess_config(self, config: types.Config) -> types.Config:
        return config

    def load_model(self) -> "FastLanguageModel": # type: ignore # noqa: F821
        if self._buffer_model is None or self._buffer_tokenizer is None:
            from unsloth import FastLanguageModel, get_chat_template

            model, tokenizer = FastLanguageModel.from_pretrained(
                **self.config.inference.model_kwargs
            )
            model = FastLanguageModel.get_peft_model(model, **self.config.inference.peft_kwargs)
            FastLanguageModel.for_inference(model)
            tokenizer = get_chat_template(tokenizer, self.chat_template)
            self._buffer_model = model
            self._buffer_tokenizer = tokenizer
        return self._buffer_model

    def load_tokenizer(self) -> "FastLanguageModel": # type: ignore # noqa: F821
        if self._buffer_model is None or self._buffer_tokenizer is None:
            from unsloth import FastLanguageModel, get_chat_template

            model, tokenizer = FastLanguageModel.from_pretrained(
                **self.config.inference.model_kwargs
            )
            model = FastLanguageModel.get_peft_model(model, **self.config.inference.peft_kwargs)
            FastLanguageModel.for_inference(model)
            tokenizer = get_chat_template(tokenizer, self.chat_template)
            self._buffer_model = model
            self._buffer_tokenizer = tokenizer
        return self._buffer_tokenizer

    def preprocess(self, text: str) -> types.Tensor:
        messages = [
            {
                "role": "user",
                "content": text,
            }
        ]

        if self.config.inference.system_prompt:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": self.config.inference.system_prompt,
                },
            )
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        return inputs

    def postprocess(self, outputs: types.Tensor) -> str:
        return self.tokenizer.decode(outputs, skip_special_tokens=True)

    def predict(self, text: str) -> str:
        inputs = self.preprocess(text)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs, **self.config.inference.inference_kwargs
            )
        prediction = self.postprocess(outputs[0][inputs.shape[1] :])
        return prediction


class LlamaOutlinesInference(BaseOutlinesInference):
    def __init__(self, config: types.Config) -> None:
        self.config = self.postprocess_config(config)
        super().__init__(config)

        self._buffer_model = None

    def postprocess_config(self, config: types.Config) -> types.Config:
        return config

    def load_model(self):
        import outlines

        if self._buffer_model is None:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self._buffer_model = outlines.models.transformers(
                self.config.inference.model_kwargs.model_name,
                device=self.device,
                model_kwargs={"torch_dtype": dtype},
            )
        return self._buffer_model

    def load_tokenizer(self):
        return True # so it's not None

    def preprocess(self, text: str) -> str:
        return text

    def postprocess(self, outputs: str) -> str:
        return outputs

    def predict(self, text: str) -> str:
        import outlines

        prompt = self.preprocess(text)


        classes: list[str] = self.config.inference.classes

        generator = outlines.generate.choice(self.model, classes)

        answer = generator(prompt)

        prediction = self.postprocess(answer)
        return prediction
