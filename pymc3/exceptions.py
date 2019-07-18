__all__ = ['SamplingError', 'IncorrectArgumentsError', 'TraceDirectoryError', 'ShapeError']


class SamplingError(RuntimeError):
    pass


class IncorrectArgumentsError(ValueError):
    pass

class TraceDirectoryError(ValueError):
    '''Error from trying to load a trace from an incorrectly-structured directory,'''
    pass

class ShapeError(ValueError):
    pass
