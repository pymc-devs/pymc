__all__ = [
    "SamplingError",
    "IncorrectArgumentsError",
    "TraceDirectoryError",
    "ImputationWarning",
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
