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

from supertrainer.inferences.llama import LlamaInference, LlamaOutlinesInference
from supertrainer.utils.deprecation import deprecated


@deprecated(
    "This module will be deprecated, please use specific `qwen` version instead (e.g. `qwen-2.5`)",
    alternative="Qwen25Inference",
)
class QwenInference(LlamaInference):
    # Generally, everything is able to be used on the parent class without modification here
    def __init__(self, config):
        super().__init__(config)
        self.chat_template = "qwen-2.5"

class QwenOutlinesInference(LlamaOutlinesInference):
    # Generally, everything is able to be used on the parent class without modification here
    pass
