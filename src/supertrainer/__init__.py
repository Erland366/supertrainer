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
DEFAULT_PATH = str(Path.home() / ".supertrainer")

if SUPERTRAINER_ROOT not in os.environ:
    os.environ[SUPERTRAINER_ROOT] = DEFAULT_PATH

supertrainer_root = Path(os.environ[SUPERTRAINER_ROOT])

if not (supertrainer_root.exists() and os.access(supertrainer_root, os.W_OK)):
    home_path = Path.home() / ".supertrainer"
    supertrainer_root = home_path
    os.environ[SUPERTRAINER_ROOT] = str(supertrainer_root)

supertrainer_root.mkdir(parents=True, exist_ok=True)

__all__ = ["logger", "StrictDict"]
