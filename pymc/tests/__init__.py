import warnings

try:
    from numpy.testing import Tester

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        test = Tester().test

except ImportError:
    warnings.warn(
        'NumPy 1.2 and nose are required to run the test suite.',
        ImportWarning)

    def test():
        return "Please install nose to run the test suite."
