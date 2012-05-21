from __future__ import with_statement
import warnings

try:
    from numpy.testing import Tester
    import numpy
    old_settings = numpy.seterr(divide = 'raise', invalid = 'raise')
    
    # TODO: Restore this implementation in 2.2, when minimum requirements are changed to Python 2.6
    # with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         test = Tester().test
    
    # Taken from http://stackoverflow.com/questions/2059675/catching-warnings-pre-python-2-6
    original_filters = warnings.filters[:]

    # Ignore warnings.
    warnings.simplefilter("ignore")

    try:
        # Execute the code that presumably causes the warnings.
        test = Tester().test

    finally:
        # Restore the list of warning filters.
        warnings.filters = original_filters
        numpy.seterr(**old_settings)
        
except ImportError:
    warnings.warn('NumPy 1.2 and nose are required to run the test suite.', ImportWarning)
    def test():
        return "Please install nose to run the test suite."