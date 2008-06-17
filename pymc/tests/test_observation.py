from numpy.testing import *
from pymc.gp import *
from test_mean import M, x
from test_cov import C
from numpy import *
from copy import copy

M = copy(M)
C = copy(C)

# Impose observations on the GP
class test_observation(TestCase):
    def test(self):
        
        obs_x = array([-.5,.5])
        V = array([.002,.002])
        data = array([3.1, 2.9])
        observe(M=M, 
                C=C,
                obs_mesh=obs_x,
                obs_V = V, 
                obs_vals = data)

        # Generate realizations

        for i in range(3):
            f = Realization(M, C)
            f(x)
            
