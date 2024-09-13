from datasets import DatasetDict
from supertrainer import logger, types
from supertrainer.data.base import BaseDataset


class LLMDataset(BaseDataset):
    def __init__(self, config: types.Config, is_testing: bool = True) -> None:
        super().__init__(config, is_testing)

    def formatting_prompt_func(
        self,
        examples: list[types.Conversation],
        use_default_system_prompt: bool = True,
        is_test_dataset: bool = True,
    ) -> dict[str, str]:
        assert "input" in examples, "Missing input key"
        assert "output" in examples, "Missing output key"

        if "instruction" not in examples:
            if use_default_system_prompt:
                logger.warning(
                    "Cannot find instruction key, using default system prompt. "
                    "If you don't want this and actually do not want to use instruction, "
                    "set `use_default_system_prompt `to False"
                )
                system_prompt = "You are an helpful AI assistant"
                insts = examples.get("instruction", [system_prompt] * len(examples["input"]))
            else:
                insts = [None] * len(examples["input"])
        else:
            insts = examples["instruction"]

        inpts = examples["input"]
        outps = examples["output"]
        texts = []
        for inst, inp, outp in zip(insts, inpts, outps):
            if inst is not None:
                chat = [
                    {"role": "system", "content": f"{inst}"},
                ]
            else:
                chat = []

            curr_chat = [
                {"role": "user", "content": f"{inp}"},
            ]
            if not is_test_dataset:
                curr_chat.append({"role": "assistant", "content": f"{outp}"})
            chat.extend(curr_chat)

            if is_test_dataset:
                logger.debug(
                    "Remove assistant output from test dataset" "instead, add the generation prompt"
                )
                text = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            else:
                text = self.tokenizer.apply_chat_template(chat, tokenize=False)
            texts.append(text)
        return {
            "text": texts,
        }

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
