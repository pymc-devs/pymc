from numpy.testing import NumpyTest

import PyMC2.tests
# import tests

__all__=['test']
def test():
    for m in PyMC2.tests.__modules__:
        print 'Testing ' + m
        mod = 'tests.%s'%m
        print mod
        testsuite = NumpyTest(mod)
        testsuite.test()

if __name__=='__main__':
    test()
    
