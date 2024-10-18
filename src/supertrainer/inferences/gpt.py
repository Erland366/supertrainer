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

from __future__ import annotations

import json
import os
from functools import partial
from typing import Any

from openai import OpenAI

from supertrainer import SUPERTRAINER_PUBLIC_ROOT, SUPERTRAINER_ROOT, logger, types
from supertrainer.inferences.base import BaseInferenceProprietary


class GPTInference(BaseInferenceProprietary):
    def __init__(self, config: types.Config) -> None:
        self.config = self.postprocess_config(config)
        super().__init__(config)

    def postprocess_config(self, config: types.Config) -> types.Config:
        return config

    def load_model(self):
        self.client = OpenAI()
        model = partial(self.client.chat.completions.create, **self.config.inference.client_kwargs)
        return model

    def load_tokenizer(self):
        return True  # Just to make it not None

    def preprocess(self, text: str) -> types.Tensor:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        if self.config.inference.system_prompt:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.config.inference.system_prompt}],
                },
            )

        return messages

    def postprocess(self, outputs: dict[str, str]) -> str:
        return outputs.choices[0].message.content

    def predict(self, text: str) -> str:
        inputs = self.preprocess(text)
        outputs = self.model(messages=inputs)
        return self.postprocess(outputs)

    def batch_predict(self, dataset: "Dataset") -> Any:  # type: ignore # noqa: F821
        batch_input_file_path = os.path.join(
            os.environ[SUPERTRAINER_ROOT], "dataset", f"{self.config.inference.batch_name}.jsonl"
        )

        os.makedirs(os.path.dirname(batch_input_file_path), exist_ok=True)

        if self.config.get("batch_id_file", None):
            batch_status = self.check_batch_status(self.config.batch_id_file)
            if batch_status.status == "completed":
                output_file_id = batch_status["output_file_id"]
                output_path = os.path.join(
                    os.environ[SUPERTRAINER_PUBLIC_ROOT],
                    "output",
                    f"{self.config.inference.batch_name}.jsonl",
                )

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                self.download_batch_output(output_file_id, output_path)
                logger.info(f"Batch completed. Output saved to {output_path}")
                return f"Batch completed. Output saved to {output_path}"
            else:
                logger.info(f"Batch status: {batch_status.status}")
                return f"Batch status: {batch_status.status}"

        if not os.path.exists(batch_input_file_path):
            system_prompt = self.config.inference.system_prompt
            output = self.generate_chat_requests(
                dataset["text"], system_prompt, self.config.inference.client_kwargs
            )

            with open(batch_input_file_path, "w") as f:
                f.write(output)

            logger.info(f"Batch input file saved to {batch_input_file_path}")

            batch = self.create_batch_request(batch_input_file_path)
            self.config.batch_id = batch.id
            logger.info(f"Batch request created with ID: {self.config.batch_id}")
            return f"Batch request created with ID: {self.config.batch_id}"

    def generate_chat_requests(
        self, messages: list[str], system_prompt: str, client_kwargs: dict
    ) -> str:
        output_lines = []

        logger.warning_once(
            "Batch request is detected. For now we only support batch requests "
            "that is using the same system prompt!"
        )

        for i, message in enumerate(messages, start=1):
            data = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": client_kwargs.get("model", "gpt-3.5-turbo-0125"),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message},
                    ],
                    "max_tokens": client_kwargs.get("max_tokens", 1000),
                    "temperature": client_kwargs.get("temperature", 1.0),
                    "top_p": client_kwargs.get("top_p", 1.0),
                },
            }
            output_lines.append(json.dumps(data))

        return "\n".join(output_lines)

    def create_batch_request(self, file_path: str):
        # TODO: We still need to use self.client, whereas I want to use self.model instead
        # Thinking of splitting the class here .-.
        self.load_model() # For loading the self.client
        batch_input_file = self.client.files.create(file=open(file_path, "rb"), purpose="batch")
        batch_input_file_id = batch_input_file.id

        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            **self.config.inference.batch_kwargs,
        )
        return batch

    def check_batch_status(self, batch_id: str):
        # TODO: We still need to use self.client, whereas I want to use self.model instead
        # Thinking of splitting the class here .-.
        self.load_model() # For loading the self.client
        return self.client.batches.retrieve(batch_id)

    def download_batch_output(self, output_file_id: str, output_path: str):
        # TODO: We still need to use self.client, whereas I want to use self.model instead
        # Thinking of splitting the class here .-.
        self.load_model() # For loading the self.client
        file_response = self.client.files.content(output_file_id)
        with open(output_path, "w") as output_file:
            output_file.write(file_response.text)

    def cancel_batch(self, batch_id: str):
        # TODO: We still need to use self.client, whereas I want to use self.model instead
        # Thinking of splitting the class here .-.
        self.load_model() # For loading the self.client
        self.client.batches.cancel(batch_id)
