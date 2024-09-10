import pytest
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig

from preskripsi_training.data.preskripsi import PreskripsiBERTDataset, PreskripsiDataset

from ..utils import print_in_test


@pytest.fixture
def preskripsi_dataset():
    # TODO: Adapt with config
    return PreskripsiDataset(
        dataset_name_or_path="masa-research/manual_absa_annotation__dgv__20240903_103",
        tokenizer_name_or_path="Qwen/Qwen2-0.5B-Instruct",
        is_testing=True,
    )


@pytest.fixture
def preskripsi_bert_dataset():
    return PreskripsiBERTDataset(
        config=DictConfig(
            {
                "dataset_kwargs": dict(
                    path="masa-research/manual_absa_annotation__dgv__20240903_103",
                    tokenizer_name_or_path="bert-base-uncased",
                )
            }
        ),
        is_testing=True,
    )


@pytest.mark.very_slow
def test_preskripsi_bert_dataset(preskripsi_bert_dataset):
    dataset = preskripsi_bert_dataset.dataset
    print_in_test(dataset)
    assert isinstance(dataset, DatasetDict) or isinstance(dataset, Dataset)


@pytest.mark.very_slow
def test_preskripsi_bert_prepare_dataset(preskripsi_bert_dataset):
    dataset = preskripsi_bert_dataset.prepare_dataset()
    assert dataset.keys() == {"train", "validation", "test"}
    assert isinstance(dataset, DatasetDict)


@pytest.mark.very_slow
def test_preskripsi_dataset(preskripsi_dataset):
    dataset = preskripsi_dataset.dataset
    print_in_test(dataset)
    assert isinstance(dataset, DatasetDict) or isinstance(dataset, Dataset)


@pytest.mark.very_slow
def test_preskripsi_split_dataset(preskripsi_dataset):
    dataset = preskripsi_dataset.dataset
    split_dataset = preskripsi_dataset.split_dataset(dataset)
    print_in_test(split_dataset)
    assert isinstance(split_dataset, DatasetDict)


@pytest.mark.very_slow
def test_preskripsi_test_tokenization(preskripsi_dataset):
    dataset = preskripsi_dataset.dataset
    split_dataset = preskripsi_dataset.split_dataset(dataset)
    formatted_dataset = preskripsi_dataset.format_for_aspect_sentiment_analysis(split_dataset)
    dataset = preskripsi_dataset.format_dataset_for_lm(formatted_dataset)
    preskripsi_dataset.test_tokenization(dataset)
    assert True


@pytest.mark.very_slow
def test_preskripsi_prepare_dataset(preskripsi_dataset):
    dataset = preskripsi_dataset.prepare_dataset()
    assert dataset.keys() == {"train", "validation", "test"}
    assert isinstance(dataset, DatasetDict)
    assert dataset["train"][0]["text"].endswith("<|im_end|>\n")
    assert dataset["validation"][0]["text"].endswith("<|im_end|>\n")
    assert dataset["test"][0]["text"].endswith("<|im_start|>assistant\n")
