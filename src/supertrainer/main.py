import os
import sys

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from supertrainer import StrictDict, logger
from supertrainer.utils import import_class, login_hf, login_wandb, memory_stats

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
    login_wandb()
    memory_stats()

    # temporary, will move this!
    classes = cfg.trainer.classes
    num_classes = len(classes)
    class2id = {class_: i for i, class_ in enumerate(classes)}
    id2class = {i: class_ for i, class_ in enumerate(classes)}
    cfg.dataset.class2id = class2id
    cfg.dataset.id2class = id2class
    cfg.dataset.num_classes = num_classes

    dataset = import_class(cfg.dataset.class_name)(cfg.dataset)
    dataset = dataset.prepare_dataset()

    # TODO: This is a hacky way to pass the dataset config
    cfg.trainer.dataset_config = cfg.dataset

    trainer = import_class(cfg.trainer.class_name)(cfg.trainer, dataset)
    trainer.train()


if __name__ == "__main__":
    main()
