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

# This file intentionally left blank
__version__ = "0.0.1"


import os
from pathlib import Path

from .utils.dict import StrictDict
from .utils.logger import logger

SUPERTRAINER_ROOT = "SUPERTRAINER_ROOT"
SUPERTRAINER_PUBLIC_ROOT = "SUPERTRAINER_PUBLIC_ROOT"

def initialize_supertrainer_root(root_env_var, default_subdir):
    default_path = str(Path.home() / default_subdir)

    if root_env_var not in os.environ:
        os.environ[root_env_var] = default_path

    root_path = Path(os.environ[root_env_var])

    if not (root_path.exists() and os.access(root_path, os.W_OK)):
        home_path = Path.home() / default_subdir
        root_path = home_path
        os.environ[root_env_var] = str(root_path)

    root_path.mkdir(parents=True, exist_ok=True)
    return root_path

supertrainer_root = initialize_supertrainer_root(SUPERTRAINER_ROOT, ".supertrainer")
supertrainer_root = initialize_supertrainer_root(SUPERTRAINER_PUBLIC_ROOT, "supertrainer")

__all__ = ["logger", "StrictDict"]
