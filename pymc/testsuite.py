from numpy.testing import NumpyTest
from pymc import tests
import warnings
import os

# Changeset
# 19/03/2007 -DH- The modules in tests/ are now imported when test is called. 

# TODO: 

warnings.simplefilter('ignore', FutureWarning)

__all__=['test']
def test():
    try:
        os.mkdir('test_results')
    except:
        pass
    os.chdir('test_results')
    testmod = __import__('pymc.tests', [], [], tests.__modules__)
    all_test_modules = tests.__modules__
    test_mod = __import__('pymc.tests', fromlist=all_test_modules)
    for m in all_test_modules:
        print 'Testing ', m, ' ...'
        NumpyTest(getattr(test_mod, m)).test(all=False)
    os.chdir('..')

if __name__=='__main__':
    test()
    
    
