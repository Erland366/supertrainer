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

import json
import os
import subprocess
from functools import partial

from anthropic import Anthropic

from supertrainer import SUPERTRAINER_PUBLIC_ROOT, logger, type_hinting
from supertrainer.inferences.base import BaseInferenceProprietary


class SonnetInference(BaseInferenceProprietary):
    def __init__(self, config: type_hinting.Config) -> None:
        self.config = self.postprocess_config(config)
        super().__init__(config)

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        return config

    def load_model(self):
        self.client = Anthropic()
        system = self.config.inference.get("system_prompt", None)

        if system:
            model = partial(
                self.client.messages.create, system=system, **self.config.inference.client_kwargs
            )
        else:
            model = partial(self.client.messages.create, **self.config.inference.client_kwargs)
        return model

    def load_tokenizer(self):
        return True  # Just to make it not None

    def preprocess(self, text: str) -> type_hinting.Tensor:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        return messages

    def postprocess(self, outputs: dict[str, str]) -> str:
        return outputs.content[0].text

    def predict(self, text: str) -> str:
        inputs = self.preprocess(text)
        outputs = self.model(messages=inputs)
        return self.postprocess(outputs)


class SonnetInstructorInference(BaseInferenceProprietary):
    def __init__(self, config: type_hinting.Config) -> None:
        self.config = self.postprocess_config(config)
        super().__init__(config)

    def postprocess_config(self, config: type_hinting.Config) -> type_hinting.Config:
        return config

    def load_model(self):
        import instructor

        system = self.config.inference.get("system_prompt", None)

        self.client = instructor.from_anthropic(Anthropic())
        if system:
            model = partial(
                self.client.messages.create, system=system, **self.config.inference.client_kwargs
            )
        else:
            model = partial(self.client.messages.create, **self.config.inference.client_kwargs)
        return model

    def load_tokenizer(self):
        return True  # Just to make it not None

    def preprocess(self, text: str) -> type_hinting.Tensor:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        return messages

    def postprocess(self, outputs: dict[str, str]) -> str:
        return outputs

    def predict(self, text: str) -> str:
        inputs = self.preprocess(text)

        from typing import Literal

        from pydantic import BaseModel, Field

        class ClassificationResponse(BaseModel):
            classes: Literal[tuple(self.config.inference.classes)]
            reasoning: str = Field(
                ...,
                description="The reasoning of why the model made the prediction of the class.",
            )

        outputs = self.model(messages=inputs, response_model=ClassificationResponse)
        return self.postprocess(outputs)

    def batch_predict(self, dataset: "Dataset") -> str:  # type: ignore # noqa: F821
        from typing import Literal

        from instructor.batch import BatchJob
        from pydantic import BaseModel, Field

        batch_input_file_path = os.path.join(
            os.environ["SUPERTRAINER_ROOT"], "dataset", f"{self.config.inference.batch_name}.jsonl"
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

            def get_messages(dataset):
                for row in dataset:
                    yield [
                        {"role": "system", "content": self.config.inference.system_prompt},
                        {"role": "user", "content": row["text"]},
                    ]

            class ClassificationResponse(BaseModel):
                classes: Literal[tuple(self.config.inference.classes)]
                reasoning: str = Field(
                    ...,
                    description="The reasoning of why the model made the prediction of the class.",
                )

            BatchJob.create_from_messages(
                messages_batch=get_messages(dataset),
                model=self.config.inference.client_kwargs.get("model", "gpt-4o"),
                file_path=batch_input_file_path,
                response_model=ClassificationResponse,
            )

            logger.info(f"Batch input file saved to {batch_input_file_path}")

        # Create batch request using subprocess
        subprocess.run(
            ["instructor", "batch", "create-from-file", "--file-path", batch_input_file_path],
            check=True,
        )

        return f"Batch request created with file path: {batch_input_file_path}"

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
        subprocess.run(
            ["instructor", "batch", "create-from-file", "--file-path", file_path], check=True
        )

        logger.info(f"Batch request created with file path: {file_path}")
        return f"Batch request created with file path: {file_path}"

    def check_batch_status(self, batch_id: str) -> str:
        import re

        result = subprocess.run(
            ["instructor", "batch", "list", "--limit", "9"],
            check=True,
            capture_output=True,
            text=True,
        )

        batch_info_pattern = re.compile(rf"{batch_id}\s+\|\s+\S+\s+\|\s+(\w+)\s+\|")
        match = batch_info_pattern.search(result.stdout)
        if match:
            return f"Batch status: {match.group(1)}"
        else:
            return "Batch ID not found"

    def cancel_batch(self, batch_id: str):
        subprocess.run(["instructor", "batch", "cancel", "--batch-id", batch_id], check=True)
        logger.info(f"Batch {batch_id} is cancelled")

    def download_batch_output(self, batch_id: str, output_path: str):
        subprocess.run(
            [
                "instructor",
                "batch",
                "download-file",
                "--download-file-path",
                output_path,
                "--batch-id",
                batch_id,
            ],
            check=True,
        )
        logger.info(f"Batch output saved to {output_path}")
