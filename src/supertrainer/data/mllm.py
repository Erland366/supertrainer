from typing import Any

import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from supertrainer import logger, types


class MLLMDatasetLoader(TorchDataset):
    def __init__(self, dataset: Dataset, config: types.Config):
        self.config = config
        self.dataset = dataset
        self.transform: None = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]
        image = sample[self.config.image_col]

        if isinstance(image, str):  # If image is a file path
            image = Image.open(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            # TODO: Change the self.user_col part
            "qa": [{"question": "What do you see?", "answer": sample[self.config.assistant_col]}],
        }

    def show_image(self, idx):
        sample = self.__getitem__(idx)
        image = sample["image"]
        if isinstance(image, Image.Image):
            plt.imshow(image)
            plt.axis("off")
            plt.show()
        else:
            raise TypeError("The image is not a PIL Image")


# TODO: Create basemodel for this
class MLLMDataset:
    def __init__(self, config: dict[str, Any], is_testing: bool = False) -> None:
        self.config = config
        self.is_testing = is_testing
        self._dataset = None

    @property
    def dataset(self) -> Dataset | DatasetDict:
        if self._dataset is None:
            self._dataset = load_dataset(self.config.dataset_kwargs.path)
            if isinstance(self._dataset, DatasetDict):
                dataset_dict = DatasetDict()
                for split in self._dataset.keys():
                    if self.is_testing:
                        dataset_pick = self._dataset[split].select(range(10))
                    else:
                        dataset_pick = self._dataset[split]
                    dataset_dict[split] = MLLMDatasetLoader(dataset_pick, self.config)
                self._dataset = dataset_dict
            else:
                self._dataset = DatasetDict(
                    {"train": MLLMDatasetLoader(self._dataset, self.config)}
                )

        return self._dataset

    def prepare_dataset(self) -> MLLMDatasetLoader:
        # TODO: This is a bug!
        logger.debug(f"Dataset prepared: {self.dataset}")
        return self.dataset
