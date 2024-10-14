import pytest

from supertrainer.data.mllm import MLLMDataset

from ..utilities import hydra_config_context, print_in_test


@pytest.fixture
def mllm_dataset():
    with hydra_config_context(
        overrides=["+experiment=train_mllm", "mllm/dataset@dataset=supertrainer_moondream"]
    ) as cfg:
        return MLLMDataset(config=cfg.dataset)


@pytest.mark.very_slow
def test_mllm_dataset(mllm_dataset):
    print_in_test(mllm_dataset.dataset)
