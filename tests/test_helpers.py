from unittest.mock import Mock, patch

import pytest
from datasets import Dataset, DatasetDict

from preskripsi_training.utils.helpers import find_max_tokens


@pytest.fixture
def mock_dataset():
    data = {
        "instruction": ["instr1", "instr2"],
        "input": ["input1", "input2"],
        "output": ["output1", "output2"],
        "text": ["instr1 input1 output1", "instr2 input2 output2"],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.tokenize.side_effect = lambda text: text.split()
    return tokenizer


def test_find_max_tokens(mock_dataset, mock_tokenizer):
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("datasets.load_dataset", return_value=DatasetDict({"train": mock_dataset})):
            max_tokens = find_max_tokens("mock_dataset", "mock_tokenizer", is_chat_formatted=True)
            assert max_tokens == 3

            max_tokens = find_max_tokens("mock_dataset", "mock_tokenizer", is_chat_formatted=False)
            assert max_tokens == 3
