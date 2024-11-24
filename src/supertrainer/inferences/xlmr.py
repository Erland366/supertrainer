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

        peft_config = PeftConfig.from_pretrained(self.config.inference.model_name)
        with self.config.allow_modification():
            # This need to be hardcoded for XLMR!
            self.config.device_map = None
            self.config.low_cpu_mem_usage = False


        model = load_model_with_adaptive_attention(
            XLMRobertaForSequenceClassification.from_pretrained,
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
            )
            breakpoint()

        model.eval()
        return model
