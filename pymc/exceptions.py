#   Copyright 2023 The PyMC Developers
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


class SamplingError(RuntimeError):
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
    """Exception for errors generated from truncated graphs"""


class NotConstantValueError(ValueError):
    pass


class BlockModelAccessError(RuntimeError):
    pass


class ParallelSamplingError(Exception):
    def __init__(self, message, chain):
        super().__init__(message)
        self._chain = chain


class RemoteTraceback(Exception):
    def __init__(self, tb):
        self.tb = tb

    def __str__(self):
        return self.tb


class VariationalInferenceError(Exception):
    """Exception for VI specific cases"""


class NotImplementedInference(VariationalInferenceError, NotImplementedError):
    """Marking non functional parts of code"""


class ExplicitInferenceError(VariationalInferenceError, TypeError):
    """Exception for bad explicit inference"""


class ParametrizationError(VariationalInferenceError, ValueError):
    """Error raised in case of bad parametrization"""


class GroupError(VariationalInferenceError, TypeError):
    """Error related to VI groups"""


class IntegrationError(RuntimeError):
    pass


class PositiveDefiniteError(ValueError):
    def __init__(self, msg, idx):
        super().__init__(msg)
        self.idx = idx
        self.msg = msg

    def __str__(self):
        return f"Scaling is not positive definite: {self.msg}. Check indexes {self.idx}."


class ParameterValueError(ValueError):
    """Exception for invalid parameters values in logprob graphs"""
