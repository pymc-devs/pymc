import unittest
import warnings
import os

# Changeset
# 19/03/2007 -DH- The modules in tests/ are now imported when test is called. 
# June 17, 2008 -DH- Switched to unittests since NumpyTest made way to nose framework.
# Sept 15, 2008 Switching to nose to run the test suite. 
warnings.simplefilter('ignore', FutureWarning)

__all__=['test']

def test():
    """Change directory to `test_results`, run the test suite and return to 
    the current directory."""
    import tests 
    try:
        os.mkdir('test_results')
    except:
        pass
    os.chdir('test_results')
    
    tests.test()
    
    os.chdir('..')

if __name__=='__main__':
    test()
    
    
