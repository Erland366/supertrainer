from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

__all__ = ["Conversation", "Config", "Dataset"]

Conversation = list[dict[str, str]]
Config = DictConfig
Dataset = Any
