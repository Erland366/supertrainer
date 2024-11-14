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
from transformers import AutoModelForCausalLM, AutoProcessor

from supertrainer import logger, type_hinting
from supertrainer.inferences.chameleon import ChameleonInference
from supertrainer.utils.helpers import load_model_with_adaptive_attention, torch_dtype


class Phi35VisionInference(ChameleonInference):
    def __init__(self, config: type_hinting.Config) -> None:
        self.config = self.postprocess_config(config)
        super().__init__(config)

        # Unsloth load model and tokenizer at the same time! Need buffer for them
        self.chat_template = "<image>Answer briefly.\n"

    def load_model(self) -> "AutoModelForCausalLM":  # type: ignore # noqa: F821
        if self._model is None:
            logger.debug("Lazy loading model")

            peft_config = PeftConfig.from_pretrained(self.config.inference.model_name)

            model = load_model_with_adaptive_attention(
                AutoModelForCausalLM.from_pretrained,
                peft_config.base_model_name_or_path,
                trust_remote_code=True,
                **self.config.inference.model_kwargs,
            )

            # patch for Phi3.5
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
            self._processor = AutoProcessor.from_pretrained(
                self.config.inference.model_name,
                trust_remote_code=True,
                **self.config.inference.processor_kwargs,
            )

            # change eos_token_id to 32000
            self._processor.tokenizer.eos_token_id = 32000

        return self._processor

    def preprocess(self, image: Image) -> type_hinting.Tensor:
        prompt_message = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{self.config.evaluation.prompt_template}",
            },
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            prompt_message, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.model.device, dtype=torch_dtype
        )
        return inputs

    def predict(self, image: Image) -> str:
        inputs = self.preprocess(image=image)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.config.inference.inference_kwargs,
                forced_eos_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                min_length=1,
            )

        # I think fault in training makes it add new line
        prediction = self.postprocess(outputs[0][inputs["input_ids"].shape[1] :]).strip()
        return prediction

    @staticmethod
    def patch_clip_for_lora(model):
        # remove unused parameters and then monkey patch
        def get_img_features(self, img_embeds):
            clip_vision_model = self.img_processor.vision_model
            hidden_states = clip_vision_model.embeddings(img_embeds)
            hidden_states = clip_vision_model.pre_layrnorm(hidden_states)
            patch_feature = clip_vision_model.encoder(
                inputs_embeds=hidden_states, output_hidden_states=True
            ).hidden_states[-1][:, 1:]
            return patch_feature

        image_embedder = model.model.vision_embed_tokens
        layer_index = image_embedder.layer_idx
        clip_layers = image_embedder.img_processor.vision_model.encoder.layers
        if layer_index < 0:
            layer_index = len(clip_layers) + layer_index
        del clip_layers[layer_index + 1 :]
        del image_embedder.img_processor.vision_model.post_layernorm
        image_embedder.get_img_features = get_img_features.__get__(image_embedder)
