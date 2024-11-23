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
from datasets import DatasetDict
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from supertrainer import StrictDict, logger
from supertrainer.utils import import_class, login_hf, memory_stats, set_global_seed

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
    memory_stats()
    set_global_seed()

    dataset = import_class(config.dataset.class_name)(config)
    dataset = dataset.prepare_dataset()

    subsets = config.dataset.dataset_kwargs.get("subsets", [None])

    with config.allow_modification():
        old_config = deepcopy(config)

    if subsets[0] is not None:
        for subset in subsets:
            if subset in config.evaluation.model_name:
                old_name = config.evaluation.model_name
                with config.allow_modification():
                    config.evaluation.model_name = config.evaluation.model_name.replace(
                        f"-{config.evaluation.subset}", ""
                    )
                logger.warning(
                    f"Subset '{config.evaluation.subset}' is in model name '{old_name}'. ",
                    "We will assume you have multiple models with the same prefix",
                    f"And we will trim the model name to '{config.evaluation.model_name}'",
                )

    for subset in subsets:
        with old_config.allow_modification():
            config = deepcopy(old_config)

        with config.allow_modification():
            config.evaluation.subset = subset
        logger.info(f"Running evaluation on subset: {subset}")

        if subset is not None:
            current_dataset = dataset[subset]

        if isinstance(dataset, DatasetDict):
            logger.warning("Multiple datasets detected. Using the test dataset by default!.")
            current_dataset = current_dataset["test"]

        evaluation = import_class(config.evaluation.class_name)(config, current_dataset)
        metrics = evaluation.evaluate()
        print(metrics)


if __name__ == "__main__":
    main()
