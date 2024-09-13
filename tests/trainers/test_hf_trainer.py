import pytest
from omegaconf import DictConfig


@pytest.fixture
def hf_bert_trainer():
    from supertrainer.trainers.hf_trainer import BERTTrainer

    return BERTTrainer(DictConfig({"classes": ["a", "b", "c"]}))


def test_config(hf_bert_trainer):
    assert hf_bert_trainer.config.num_classes == 3
    assert hf_bert_trainer.config.class2id == {"a": 0, "b": 1, "c": 2}
    assert hf_bert_trainer.config.id2class == {0: "a", 1: "b", 2: "c"}
