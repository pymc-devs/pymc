from numpy.testing import NumpyTest

from PyMC2 import tests
#import tests

__all__=['test']
def test():
    for m in tests.__modules__:
        print 'Testing ' + m
        mod = 'tests.%s'%m
        print mod
        testsuite = NumpyTest(mod)
        testsuite.test()

if __name__=='__main__':
    test()
    
