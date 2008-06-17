from numpy.testing import *
from numpy import *
from pymc.gp import Mean
from copy import copy

def quadfun(x, a, b, c):
    return (a * x ** 2 + b * x + c)

M = Mean(eval_fun = quadfun, a = 1., b = .5, c = 2.)

def constant(x, val):
    """docstring for parabolic_fun"""
    return zeros(x.shape[:-1],dtype=float) + val
    
N = Mean(constant, val = 0.)

x=arange(-1,1,.01)
y=vstack((x,x)).T

class test_mean(TestCase):
    def test(self):
        assert(M(x).shape == x.shape)
        assert(N(y).shape == (y.shape[0],))
