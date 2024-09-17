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

from datasets import DatasetDict

from supertrainer import logger, types
from supertrainer.data.base import BaseDataset


class LLMDataset(BaseDataset):
    """This class is made to support Instruction Question Answering Type of Dataset!

    Each row is expected to have 3 columns:
        - Instruction
        - Input
        - Output
    """

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


class ConversationLLMDataset(BaseDataset):
    """This class is made to support Conversation type of Dataset like ShareGPT!"""

    def __init__(self, config: types.Config, is_testing: bool = True) -> None:
        super().__init__(config, is_testing)

    @staticmethod
    def convert_sharegpt_to_chatml(example) -> None:
        """
        Converts a conversation example from the ShareGPT format to the ChatML format.
        Args:
            example (dict): The conversation example in the ShareGPT format.
        Returns:
            dict: The conversation example in the ChatML format.
        """
        new_conversation = []
        for convo in example["conversations"]:
            new_convo = {}
            if convo["from"] == "system":
                new_convo["role"] = "system"
            elif convo["from"] == "human":
                new_convo["role"] = "user"
            elif convo["from"] == "gpt":
                new_convo["role"] = "assistant"

            new_convo["content"] = convo["value"]

            new_conversation.append(new_convo)

        example["conversations"] = new_conversation

        return example

    def formatting_prompt_func(
        self,
        examples: list[types.Conversation],
        use_default_system_prompt: bool = True,
        is_test_dataset: bool = True,
    ) -> dict[str, str]:
        """Batch formatting right here"""
        assert "conversations" in examples, "Missing conversations key"

        conversations = examples["conversations"]

        if "from" in conversations[0][0]:
            # This means it's in ShareGPT format

            new_conversations = []
            for convos in conversations:
                new_convos = []
                for convo in convos:
                    new_convo = {}
                    if convo["from"] == "system":
                        new_convo["role"] = "system"
                    elif convo["from"] == "human":
                        new_convo["role"] = "user"
                    elif convo["from"] == "gpt":
                        new_convo["role"] = "assistant"

                    new_convo["content"] = convo["value"]

                    new_convos.append(new_convo)
                new_conversations.append(new_convos)

        conversations = new_conversations

        texts = []
        for convos in conversations:
            if use_default_system_prompt and "system" not in convos[0].values():
                convos.insert(0, {"role": "system", "content": "You are an helpful AI assistant"})
            if is_test_dataset:
                logger.debug(
                    "Remove assistant output from test dataset "
                    "instead, add the generation prompt"
                )
                convos = convos[:-1]
                text = self.tokenizer.apply_chat_template(
                    convos, tokenize=False, add_generation_prompt=True
                )
            else:
                text = self.tokenizer.apply_chat_template(convos, tokenize=False)
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

    def prepare_dataset(self) -> types.Any:
        logger.debug("Preparing dataset")
        dataset = self.dataset

        if not (all(x in ["train", "validation", "test"] for x in dataset)):
            logger.debug("Splitting dataset")
            dataset = self.split_dataset(dataset)

            logger.debug(f"Dataset split: {dataset}")

        logger.debug("Applying formatting prompts")
        dataset = self.format_dataset(dataset)

        self.test_tokenization(dataset)

        # Print some examples from the dataset to inspect the tokenizer's output
        print("*** Example from the dataset ***")
        for i in range(5):
            print(f"Example {i+1}:")
            self.print_text_after_substring(dataset["train"][i]["text"], "<|end_of_text|>")
            print("-" * 20)

        logger.debug("Dataset preparation completed")

        return dataset
