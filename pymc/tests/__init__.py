# Changeset
# 19/03/2007 -DH- Commented modules import. They are now imported by testsuite.
# 15/10/2008 -DH- Testing is now done through nose.
from __future__ import with_statement
import warnings
# warnings.simplefilter('default', ImportWarning)
try:
    from numpy.testing import Tester
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        test = Tester().test
except ImportError:
    warnings.warn('NumPy 1.2 and nose are required to run the test suite.', ImportWarning)
    def test():
        return "Please install nose to run the test suite."


