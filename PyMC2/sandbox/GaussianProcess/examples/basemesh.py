# GaussianProcess/examples/cov.py

from GaussianProcess import *
from GaussianProcess.cov_funs import matern
from numpy import *
from pylab import *

# x will be the base mesh
x = arange(-1.,1.,.1)


# Create a Covariance object with x as the base mesh
C = Covariance( eval_fun = matern, 
                base_mesh = x,
                diff_degree = 1.4, amp = .4, scale = .5)


# Create a Mean object. It will inherit C's base mesh, which is x.
def linfun(x, a, b, c):
    return a * x ** 2 + b * x + c

M = Mean(   eval_fun = linfun,
            base_mesh = x, 
            a = 1., b = .5, c = 2.)


# Impose observations on the GP
obs_x = array([-.5,.5])
tau = array([500.,500.])
d = array([3.1, 2.9])

observe(C=C, 
        M=M,
        obs_mesh=obs_x,
        obs_taus = tau, 
        obs_vals = d)


# # Generate realization
f = Realization(M, C)