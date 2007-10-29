from numpy.testing import NumpyTest

from PyMC2 import tests
import os
# Changeset
# 19/03/2007 -DH- The modules in tests/ are now imported when test is called. 

# TODO: 

__all__=['test']
def test():
    try:
        os.mkdir('test_results')
    except:
        pass
    os.chdir('test_results')
    testmod = __import__('PyMC2.tests', [], [], tests.__modules__)
    
    for m in tests.__modules__:
        print 'Testing ' + m
        mod = getattr(testmod, m)
        print mod
        testsuite = NumpyTest(mod)
        testsuite.test()
    
    os.chdir('..')

if __name__=='__main__':
    test()
    
    
