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
from copy import deepcopy

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import wandb
from supertrainer import StrictDict, logger
from supertrainer.utils import import_class, login_hf, login_wandb, memory_stats, set_global_seed

load_dotenv()

# Enable HF Transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stderr, level="DEBUG")


@hydra.main(config_path="../../configs/", config_name="train", version_base=None)
def main(config: DictConfig):
    # Enable editing on the omegaconf
    config = StrictDict(OmegaConf.to_container(config, resolve=True))
    login_hf()
    login_wandb()
    memory_stats()
    set_global_seed()

    if config.is_testing:
        logger.info("TESTING MODE ENABLED")

    os.environ["WANDB_PROJECT"] = config.wandb.project
    if "entity" in config.wandb and config.wandb.entity:
        os.environ["WANDB_ENTITY"] = config.wandb.entity

    dataset = import_class(config.dataset.class_name)(config)
    dataset = dataset.prepare_dataset()

    subsets = config.dataset.dataset_kwargs.get("subsets", [None])

    with config.allow_modification():
        old_config = deepcopy(config)

    for subset in subsets:
        with old_config.allow_modification():
            config = deepcopy(old_config)
        with config.allow_modification():
            config.trainer.subset = subset

        trainer = import_class(config.trainer.class_name)(config, dataset[subset])
        trainer.train()

        wandb.finish()


if __name__ == "__main__":
    main()
