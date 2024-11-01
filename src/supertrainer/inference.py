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

import os
import sys

import hydra
from datasets import DatasetDict
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from supertrainer import StrictDict, logger
from supertrainer.utils import import_class, login_hf, memory_stats, set_global_seed
from supertrainer.utils.helpers import load_dataset_plus_plus

load_dotenv()

# Enable HF Transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stderr, level="DEBUG")


@hydra.main(config_path="../../configs/", config_name="train", version_base=None)
def main(cfg: DictConfig):
    # Enable editing on the omegaconf
    cfg = StrictDict(OmegaConf.to_container(cfg, resolve=True))
    login_hf()
    memory_stats()
    set_global_seed()

    inference = import_class(cfg.inference.class_name)(cfg)
    if cfg.get("batch", False):
        dataset = load_dataset_plus_plus(cfg.inference.batch_dataset)
        if isinstance(dataset, DatasetDict):
            dataset = dataset["test"]
        inference.batch_predict(dataset)
    else:
        inference.iterative_predict()


if __name__ == "__main__":
    main()
