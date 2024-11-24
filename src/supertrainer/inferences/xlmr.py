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

from peft import PeftConfig, PeftModel
from transformers import XLMRobertaForSequenceClassification

from supertrainer import logger
from supertrainer.inferences.bert import BertInference
from supertrainer.utils.helpers import load_model_with_adaptive_attention


class XLMRInference(BertInference):
    # Generally, everything is able to be used on the parent class without modification here
    def load_model(self) -> XLMRobertaForSequenceClassification:
        """
        Load XLMRoberta model with support for three scenarios:
        1. Load full fine-tuned model directly
        2. Load base model with adapter (LoRA)
        3. Load base model only (without adapter)
        """
        logger.info(
            f"Starting XLMRoberta model loading process for: {self.config.inference.model_name}"
        )

        try:
            # Try loading as a PEFT model first
            peft_config = PeftConfig.from_pretrained(self.config.inference.model_name)
            is_peft_model = True
            base_model_name = peft_config.base_model_name_or_path
            logger.info("✓ Successfully loaded PEFT config - detected adapter-based model")
            logger.info(f"  └─ Base model path: {base_model_name}")
        except Exception as e:
            # If failed, assume it's a full model
            is_peft_model = False
            base_model_name = self.config.inference.model_name
            logger.info("→ Failed to load PEFT config - assuming full model")
            logger.debug(f"  └─ PEFT loading error: {str(e)}")

        # Apply XLMRoberta-specific configuration
        with self.config.allow_modification():
            logger.info("Applying XLMRoberta-specific configuration")
            # These need to be hardcoded for XLMR!
            self.config.device_map = None
            self.config.low_cpu_mem_usage = False
            logger.info("  └─ Set device_map: None")
            logger.info("  └─ Set low_cpu_mem_usage: False")

        if is_peft_model:
            if self.config.inference.base_only:
                logger.info("CASE 1: Loading base model only (without adapter)")
            else:
                logger.info("CASE 2: Loading base model with adapter (LoRA)")

            # Load base model first
            logger.info(f"Loading base model: {base_model_name}")
            model = load_model_with_adaptive_attention(
                XLMRobertaForSequenceClassification.from_pretrained,
                base_model_name,
                num_labels=len(self.config.inference.classes),
                trust_remote_code=True,
                **self.config.inference.model_kwargs,
            )
            logger.info("✓ Successfully loaded base model")

            # Optionally load adapter
            if not self.config.inference.base_only:
                logger.info(f"Loading PEFT adapter: {self.config.inference.model_name}")
                model = PeftModel.from_pretrained(
                    model,
                    self.config.inference.model_name,
                )
                logger.info("✓ Successfully loaded PEFT adapter")
        else:
            # Load full model directly
            logger.info("CASE 3: Loading full fine-tuned model directly")
            logger.info(f"Loading model from: {base_model_name}")
            model = load_model_with_adaptive_attention(
                XLMRobertaForSequenceClassification.from_pretrained,
                base_model_name,
                num_labels=len(self.config.inference.classes),
                trust_remote_code=True,
                **self.config.inference.model_kwargs,
            )
            logger.info("✓ Successfully loaded full model")

        model.eval()
        logger.info("Model loading completed - model set to evaluation mode")
        return model
