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

# fmt: off
import inspect
import transformers
from transformers.utils import add_start_docstrings, replace_return_docstrings, add_start_docstrings_to_model_forward
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Union, Tuple
import torch

def patch_chameleon_model():
    print("Patching ChameleonForConditionalGeneration.forward...")
    try:
        from transformers.models.chameleon.modeling_chameleon import ChameleonForConditionalGeneration
    except ImportError:
        raise ImportError("Could not import ChameleonForConditionalGeneration")


    function = inspect.getsource(eval("transformers.models.chameleon.modeling_chameleon.ChameleonForConditionalGeneration.forward"))

    # Find where the function definition starts
    where = min(function.find("def"), function.find("@")) # find level indentaton by either from def or decorator
    function = function.split("\n")
    function = "\n".join(x[where:] for x in function)

    replacer = \
    "image_tokens = self.model.vocabulary_mapping.image_tokens\n"\
    "logits[:, :, image_tokens] = torch.finfo(logits.dtype).min"

    replaced = \
    "# image_tokens = self.model.vocabulary_mapping.image_tokens\n"\
    "# logits[:, :, image_tokens] = torch.finfo(logits.dtype).min"


    replaced = replaced.split("\n")
    replaced = "\n".join(" "*where + x for x in replaced)

    replacer = replacer.split("\n")
    replacer = "\n".join(" "*where + x for x in replacer)

    function = function.replace(replacer, replaced)

    # # remove decorator
    function = function.split("\n")
    function = "\n".join(x for x in function if not x.startswith("@"))

    # Just exec it in the class's namespace
    exec(function, globals())
    exec(f"transformers.models.chameleon.modeling_chameleon.ChameleonForConditionalGeneration.forward = forward", globals())

    return "Successfully patched ChameleonForConditionalGeneration.forward!"

# Patching the model
# patch_chameleon_model()
