__all__ = ['SamplingError', 'IncorrectArgumentsError', 'TraceDirectoryError']


class SamplingError(RuntimeError):
    pass


class IncorrectArgumentsError(ValueError):
    pass

class TraceDirectoryError(ValueError):
    '''Error from trying to load a trace from an incorrectly-structured directory,'''
    pass
