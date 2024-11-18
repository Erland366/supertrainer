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

import copy
import json
from contextlib import contextmanager
from typing import Any

from addict import Dict


class StrictDict(Dict):
    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_allow_missing", False)
        object.__setattr__(self, "_allow_modifications", True)
        super().__init__()
        self.update(self._convert_dict(*args, **kwargs))
        object.__setattr__(self, "_allow_modifications", False)

    def __deepcopy__(self, memo):
        # Create a new instance with modifications allowed
        result = StrictDict()
        # Ensure we can modify the new instance during deep copy
        object.__setattr__(result, "_allow_modifications", True)

        # Deep copy all items
        for key, value in self.items():
            result[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)

        # Lock the new instance after copying
        object.__setattr__(result, "_allow_modifications", False)
        return result

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

    def __getitem__(self, key):
        str_key = str(key)
        if str_key not in self:
            raise KeyError(key)
        return super().__getitem__(str_key)

    def __setattr__(self, key, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        elif not object.__getattribute__(self, "_allow_modifications"):
            raise AttributeError(
                "StrictDict is locked. Use method 'set_value' or contextmanager "
                "'.allow_modification' to modify"
            )
        else:
            self[key] = value

    def __setitem__(self, key, value):
        if not object.__getattribute__(self, "_allow_modifications"):
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
        # Save original states
        original_states = {}

        def set_allow_modifications(d):
            original_states[id(d)] = object.__getattribute__(d, "_allow_modifications")
            object.__setattr__(d, "_allow_modifications", True)
            for value in d.values():
                if isinstance(value, StrictDict):
                    set_allow_modifications(value)

        def restore_allow_modifications(d):
            object.__setattr__(d, "_allow_modifications", original_states[id(d)])
            for value in d.values():
                if isinstance(value, StrictDict):
                    restore_allow_modifications(value)

        set_allow_modifications(self)
        try:
            yield self
        finally:
            restore_allow_modifications(self)

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
                pass  # Handle unserializable values if necessary
        return result

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, StrictDict) else v for k, v in self.items()}
