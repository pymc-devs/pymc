from numpy.testing import NumpyTest
from pymc import tests

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
    all_tests_modules = __import__('pymc.tests', [], [], 'pymc.tests.__modules__')        
    NumpyTest(all_tests_modules).test()
    os.chdir('..')

if __name__=='__main__':
    test()
    
    
