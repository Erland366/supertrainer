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

import functools
import warnings
from contextlib import ContextDecorator
from datetime import date, datetime
from typing import Any, Callable, Optional, Type, TypeVar, Union

T = TypeVar("T")


class deprecated(ContextDecorator):
    """
    A context manager and decorator that issues deprecation warnings.
    Can be used as a context manager, function decorator, or class decorator.

    Examples:
        # As a context manager
        with deprecated("This feature will be removed", remove_date="2025-01-01"):
            legacy_operation()

        # As a function decorator
        @deprecated("Use new_function() instead")
        def old_function():
            pass

        # As a class decorator
        @deprecated("This class is deprecated", alternative="NewClass")
        class OldClass:
            pass
    """

    def __init__(
        self,
        message: str,
        remove_date: Optional[str] = None,
        alternative: Optional[str] = None,
        stack_level: int = 2,
    ):
        self.message = message
        self.remove_date = remove_date
        self.alternative = alternative
        self.stack_level = stack_level

    def _format_message(self) -> str:
        parts = [self.message]

        if self.remove_date:
            try:
                removal_date = datetime.strptime(self.remove_date, "%Y-%m-%d").date()
                days_left = (removal_date - date.today()).days
                if days_left > 0:
                    parts.append(f"Will be removed on {self.remove_date} ({days_left} days left)")
                else:
                    parts.append(f"Was scheduled for removal on {self.remove_date}")
            except ValueError:
                parts.append(f"Will be removed on {self.remove_date}")

        if self.alternative:
            parts.append(f"Use {self.alternative} instead")

        return ". ".join(parts)

    def __enter__(self):
        warnings.warn(
            self._format_message(), category=DeprecationWarning, stacklevel=self.stack_level
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(
        self, obj: Union[Callable[..., Any], Type[T]]
    ) -> Union[Callable[..., Any], Type[T]]:
        """
        Allows the class to be used as a decorator for both functions and classes.
        """
        if isinstance(obj, type):
            # If decorating a class, wrap its __new__ method
            original_new = obj.__new__

            @functools.wraps(original_new)
            def wrapped_new(cls, *args, **kwargs):
                with self:
                    if original_new is object.__new__:
                        # If the class doesn't define __new__, call object.__new__ without args
                        instance = original_new(cls)
                    else:
                        # If the class defines __new__, pass all arguments
                        instance = original_new(cls, *args, **kwargs)
                return instance

            obj.__new__ = wrapped_new
            return obj
        else:
            # If decorating a function, wrap it normally
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                with self:
                    return obj(*args, **kwargs)

            return wrapper
