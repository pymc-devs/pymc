__all__ = ['SamplingError', 'IncorrectArgumentsError']


class SamplingError(RuntimeError):
    pass


class IncorrectArgumentsError(ValueError):
    pass
