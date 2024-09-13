import json
from contextlib import contextmanager
from typing import Any

from addict import Dict

from supertrainer.utils.helpers import logger


class StrictDict(Dict):
    def __init__(self, *args, **kwargs):
        self.__dict__["_allow_missing"] = False
        self.__dict__["_allow_modifications"] = (
            True  # Temporarily allow modifications during initialization
        )
        super().__init__()  # Initialize an empty Dict first
        self.update(self._convert_dict(*args, **kwargs))  # Use our custom update method
        self.__dict__["_allow_modifications"] = False  # Lock after initialization

    def _convert_dict(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            return self._dict_to_strictdict(args[0])
        elif kwargs:
            return self._dict_to_strictdict(kwargs)
        return {}

    def _dict_to_strictdict(self, d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[str(k)] = StrictDict(v)
            elif isinstance(v, list):
                result[str(k)] = [
                    StrictDict(item) if isinstance(item, dict) else item for item in v
                ]
            else:
                result[str(k)] = v
        return result

    def __getattr__(self, item):
        if item not in self:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
        return super().__getattr__(item)

    def __missing__(self, name):
        return None

    def __getitem__(self, key):
        str_key = str(key)
        if str_key not in self:
            raise KeyError(key)
        return super().__getitem__(str_key)

    def __setattr__(self, key, value):
        if not self._allow_modifications:
            raise AttributeError(
                "StrictDict is locked. Use method 'set_value' or contextmanager "
                "'.allow_modification' to modify"
            )
        super().__setattr__(str(key), value)

    def __setitem__(self, key, value):
        if not self._allow_modifications:
            raise KeyError(
                "StrictDict is locked. Use method 'set_value' or contextmanager "
                "'.allow_modification' to modify"
            )
        super().__setitem__(str(key), value)

    def set_value(self, key_path: str, value: Any):
        keys = self._parse_key_path(key_path)
        self._recursive_set(self, keys, value)

    def _parse_key_path(self, key_path: str):
        return [
            int(k) if k.isdigit() else k
            for k in key_path.replace("[", ".").replace("]", "").split(".")
        ]

    def _recursive_set(self, current_dict, keys, value):
        if len(keys) == 1:
            with current_dict.allow_modification():
                current_dict[keys[0]] = value
        else:
            if str(keys[0]) not in current_dict or not isinstance(
                current_dict[str(keys[0])], StrictDict
            ):
                with current_dict.allow_modification():
                    current_dict[keys[0]] = StrictDict()
            self._recursive_set(current_dict[str(keys[0])], keys[1:], value)

    @contextmanager
    def allow_modification(self):
        original_allow_modifications = self._allow_modifications
        self.__dict__["_allow_modifications"] = True
        try:
            yield self
        finally:
            self.__dict__["_allow_modifications"] = original_allow_modifications

    def is_serializable(self, value):
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False

    def to_serializable_dict(self):
        result = {}
        for k, v in self.items():
            if isinstance(v, StrictDict):
                result[k] = v.to_serializable_dict()
            elif self.is_serializable(v):
                result[k] = v
            else:
                logger.warning_once(
                    f"Key '{k}' contains an unserializable value of "
                    f"type {type(v).__name__}. Skipping."
                )
        return result

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, StrictDict) else v for k, v in self.items()}

    # def __repr__(self):
    #     return f"StrictDict({super().__repr__()})"
