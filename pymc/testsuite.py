from pymc import tests
import unittest
import warnings
import os

# Changeset
# 19/03/2007 -DH- The modules in tests/ are now imported when test is called. 
# June 17, 2008 -DH- Switched to unittests since NumpyTest made way to nose framework.

warnings.simplefilter('ignore', FutureWarning)

__all__=['test']
def test():
    try:
        os.mkdir('test_results')
    except:
        pass
    os.chdir('test_results')
    
    # Import all tests modules declared in pymc/tests/__init__.py
    __import__('pymc.tests', fromlist=tests.__modules__)
    
    # Create a test suite from all the tests
    L = unittest.TestLoader()
    S = L.loadTestsFromNames(tests.__modules__, tests)
    
    # Run the test suite.
    unittest.TextTestRunner(verbosity=1).run(S)
    os.chdir('..')

if __name__=='__main__':
    test()
    
    
