__all__ = [
    "SamplingError",
    "IncorrectArgumentsError",
    "TraceDirectoryError",
    "ImputationWarning",
    "ShapeError"
]


class SamplingError(RuntimeError):
    pass


class IncorrectArgumentsError(ValueError):
    pass


class TraceDirectoryError(ValueError):
    """Error from trying to load a trace from an incorrectly-structured directory,"""

    pass


class ImputationWarning(UserWarning):
    """Warning that there are missing values that will be imputed."""

    pass


class ShapeError(Exception):
    """Error that the shape of a variable is incorrect."""
    def __init__(self, message, actual=None, expected=None):
        if expected and actual:
            super().__init__('{} (actual {} != expected {})'.format(message, actual, expected))
        else:
            super().__init__(message)


class DtypeError(TypeError):
    """Error that the dtype of a variable is incorrect."""
    def __init__(self, message, actual=None, expected=None):
        if expected and actual:
            super().__init__('{} (actual {} != expected {})'.format(message, actual, expected))
        else:
            super().__init__(message)
