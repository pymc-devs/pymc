# Changeset
# 19/03/2007 -DH- Commented modules import. They are now imported by testsuite.
# 15/10/2008 -DH- Testing is now done through nose. 
import warnings
warnings.simplefilter('default', ImportWarning)
try:
    from numpy.testing import Tester
    test = Tester().test
except ImportError:
    warnings.warn('NumPy 1.2  and nose are required to run the test suite.', ImportWarning)
    def test():
        return "Please install nose to run the test suite."

    
