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

from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from supertrainer import logger, types


class BaseInference(ABC):
    def __init__(self, config: types.Config) -> None:
        self.config = self.postprocess_config(config)
        self._model = None
        self._tokenizer = None

    def postprocess_config(self, config: types.Config) -> types.Config:
        return config

    @abstractmethod
    def load_model(self) -> PreTrainedModel:
        """Load and return the model."""
        pass

    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and return the tokenizer."""
        pass

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            logger.debug("Loading model")
            self._model = self.load_model()
            self._model.eval()  # Set model to evaluation mode
            self._model.to(self.device)
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            logger.info("Loading tokenizer")
            self._tokenizer = self.load_tokenizer()

            if self._tokenizer.pad_token is None:
                self._tokenizer.add_special_tokens({"pad_token": self._tokenizer.eos_token})
            if self._tokenizer.model_max_length > 100_000:
                self._tokenizer.model_max_length = 2048
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def preprocess(self, text: str) -> types.Tensor:
        """Preprocess the text."""
        pass

    @abstractmethod
    def postprocess(self, outputs: types.Tensor) -> types.Tensor:
        """Postprocess the outputs."""
        pass

    @abstractmethod
    def predict(self, text: str) -> types.Tensor:
        """Predict the output for the given text."""
        pass

    def iterative_predict(self):
        """Run iterative inference in a loop."""
        logger.info("Starting iterative inference. Type 'exit' or 'quit' to stop.")
        try:
            while True:
                text = input("Enter input for prediction: ").strip()
                if text.lower() in {"exit", "quit"}:
                    logger.info("Stopping iterative inference.")
                    break
                if not text:
                    print("Empty input. Please enter valid text.")
                    continue
                prediction = self.predict(text)
                print(f"Prediction: {prediction}")
        except KeyboardInterrupt:
            logger.info("Iterative inference interrupted by user.")

class BaseInferenceProprietary(ABC):
    def __init__(self, config: types.Config) -> None:
        self.config = self.postprocess_config(config)
        self._model = None
        self._tokenizer = None

    def postprocess_config(self, config: types.Config) -> types.Config:
        return config

    @abstractmethod
    def load_model(self) -> PreTrainedModel:
        """Load and return the model."""
        pass

    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and return the tokenizer."""
        pass

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            logger.debug("Loading model")
            self._model = self.load_model()
        return self._model

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def preprocess(self, text: str) -> types.Tensor:
        """Preprocess the text."""
        pass

    @abstractmethod
    def postprocess(self, outputs: types.Tensor) -> types.Tensor:
        """Postprocess the outputs."""
        pass

    @abstractmethod
    def predict(self, text: str) -> types.Tensor:
        """Predict the output for the given text."""
        pass

    def iterative_predict(self):
        """Run iterative inference in a loop."""
        logger.info("Starting iterative inference. Type 'exit' or 'quit' to stop.")
        try:
            while True:
                text = input("Enter input for prediction: ").strip()
                if text.lower() in {"exit", "quit"}:
                    logger.info("Stopping iterative inference.")
                    break
                if not text:
                    print("Empty input. Please enter valid text.")
                    continue
                prediction = self.predict(text)
                print(f"Prediction: {prediction}")
        except KeyboardInterrupt:
            logger.info("Iterative inference interrupted by user.")

class BaseOutlinesInference(BaseInferenceProprietary):
    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            logger.info("Loading tokenizer")
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer
