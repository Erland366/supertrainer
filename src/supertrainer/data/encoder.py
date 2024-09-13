from datasets import DatasetDict
from supertrainer import logger, types
from supertrainer.data.base import BaseDataset


class EncoderDataset(BaseDataset):
    def __init__(self, config: types.Config, is_testing: bool = True) -> None:
        super().__init__(config, is_testing)

    def formatting_prompt_func(
        self, examples: list[types.Conversation], is_test_dataset: bool = True
    ):
        assert "input" in examples, "Missing input key"
        assert "output" in examples, "Missing output key"

        if "instruction" in examples:
            logger.warning(
                "We found instruction key, but we are not using it since we are "
                "using BERT-based model"
            )

        if is_test_dataset:
            pass

        inpts = examples["input"]
        outps = examples["output"]
        ents = examples["entity"]
        texts = []
        for ent, inp in zip(ents, inpts):
            text = f"{inp}: {ent}"
            texts.append(text)
        # SHIT THIS TOOK SO LONG, APPARENTLY IT MUST BE `labels` AND CANNOT ANYTHING ELSE
        return {"text": texts, "labels": outps}

    def format_dataset(self, dataset: types.Conversation) -> types.Conversation:
        processed_dataset = DatasetDict()

        for split_name, split_dataset in dataset.items():
            is_test_dataset = split_name == "test"

            processed_split = split_dataset.map(
                lambda examples: self.formatting_prompt_func(
                    examples, is_test_dataset=is_test_dataset
                ),
                batched=True,
            )

            processed_dataset[split_name] = processed_split

        return processed_dataset
