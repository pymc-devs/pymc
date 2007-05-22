from numpy.testing import *
import numpy as np
from PyMC2.LazyFunction import LazyFunction
from PyMC2 import parameter, data, node, Normal

p = Normal('p', value=5., mu=4., tau=1.)

@node
def n(p=p):
    return p*2.

class test_cache(NumpyTestCase):
    def check_simple(self):
        L = LazyFunction(n._eval_fun, {'p':p.value}, 2)
        print L.cached_values()
        p.random()
        n.value
        print L.cached_values()
        
