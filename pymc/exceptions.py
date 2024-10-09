#   Copyright 2024 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

__all__ = [
    "SamplingError",
    "IncorrectArgumentsError",
    "TraceDirectoryError",
    "ImputationWarning",
    "ShapeWarning",
    "ShapeError",
]


class SamplingError(RuntimeError):
    pass


class IncorrectArgumentsError(ValueError):
    pass


class TraceDirectoryError(ValueError):
    """Error from trying to load a trace from an incorrectly-structured directory."""

    pass


class ImputationWarning(UserWarning):
    """Warning that there are missing values that will be imputed."""

    pass


class ShapeWarning(UserWarning):
    """Something that could lead to shape problems down the line."""

    pass


class ShapeError(Exception):
    """Error that the shape of a variable is incorrect."""

    def __init__(self, message, actual=None, expected=None):
        if actual is not None and expected is not None:
            super().__init__(f"{message} (actual {actual} != expected {expected})")
        elif actual is not None and expected is None:
            super().__init__(f"{message} (actual {actual})")
        elif actual is None and expected is not None:
            super().__init__(f"{message} (expected {expected})")
        else:
            super().__init__(message)


class DtypeError(TypeError):
    """Error that the dtype of a variable is incorrect."""

    def __init__(self, message, actual=None, expected=None):
        if actual is not None and expected is not None:
            super().__init__(f"{message} (actual {actual} != expected {expected})")
        elif actual is not None and expected is None:
            super().__init__(f"{message} (actual {actual})")
        elif actual is None and expected is not None:
            super().__init__(f"{message} (expected {expected})")
        else:
            super().__init__(message)


class TruncationError(RuntimeError):
    """Exception for errors generated from truncated graphs."""


class NotConstantValueError(ValueError):
    pass


class BlockModelAccessError(RuntimeError):
    pass


class UndefinedMomentException(Exception):
    pass
