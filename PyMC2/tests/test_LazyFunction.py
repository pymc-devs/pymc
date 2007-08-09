from numpy.testing import *
import numpy as np
from numpy.random import random
from PyMC2.PyrexLazyFunction import LazyFunction
from PyMC2 import parameter, data, node, Normal

@data
def mu(value=5.):
    return 0.
    
@data
def tau(value=1.):
    return 0.

p = Normal('p', value=5., mu=mu, tau=tau)

@node
def DumbNode(p=p):
    return random()*p

def f(p, d):
    return p**2 / d

class test_cache(NumpyTestCase):

    def check_simple(self):
        L = LazyFunction(f, {'p':p,'d':DumbNode}, 2)
        for i in range(10000):    
            last_dumbnode_value = DumbNode.value        
            p.random()
            
            # Make sure returned value is correct.
            assert_equal(L.get(), p.value**2 / DumbNode.value)
            
            # Make sure correct values got into cache.
            assert_equal(L.cached_values[0], p.value**2 / DumbNode.value)
            assert_equal(L.cached_args[0], p.value)
            assert_equal(L.argument_values['p'], p.value)
            assert_equal(L.ultimate_arg_values[0], p.value)
            
            assert_equal(L.cached_args[1], DumbNode.value)
            assert_equal(L.argument_values['d'], DumbNode.value)
            assert_equal(L.ultimate_arg_values[1], DumbNode.value)
            
            L.get()
            
            if i>1:
                
                # Make sure last values got pushed down cache.
                assert_equal(L.cached_values[1], p.last_value**2 / last_dumbnode_value)
                assert_equal(L.cached_args[2], p.last_value)
                assert_equal(L.cached_args[3], last_dumbnode_value)
                        
if __name__ == '__main__':
    NumpyTest().run()
