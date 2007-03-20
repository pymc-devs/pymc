from numpy.testing import NumpyTest

import PyMC2.tests

# Changeset
# 19/03/2007 -DH- The modules in tests/ are now imported when test is called. 

__all__=['test']
def test():
    for m in PyMC2.tests.__modules__:
        print 'Testing ' + m
        mod = __import__('PyMC2.tests.%s'%m)
        print mod
        testsuite = NumpyTest(mod)
        testsuite.test()

if __name__=='__main__':
    test()
    
