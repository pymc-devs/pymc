# GaussianProcess/examples/PyMC_observed_form.py

from GaussianProcess import *
from GaussianProcess.cov_funs import Matern
from numpy import *
from pylab import *
from PyMC2 import *


x = arange(-1.,1.,.1)
obs_x = array([-.5,.5])

# Observation precision
@parameter
def tau(value=500., alpha = 1., beta = 500.):
    """tau ~ gamma(alpha, beta)"""
    if value<=0.:
        raise LikelihoodError
    return flib.gamma(value, alpha, beta)

# Prior parameters of C
@parameter
def C_p1(value=1.4, mu=1.4, tau=1.):
    """C_p1 ~ lognormal(mu, tau)"""
    if value<=0.:
        raise LikelihoodError
    return flib.lognormal(value, mu, tau)
    
@parameter
def C_p2(value=.4, mu = .4, tau = 1.):
    """C_p2 ~ lognormal(mu, tau)"""
    if value<=0.:
        raise LikelihoodError
    return flib.lognormal(value, mu, tau)
    
@parameter
def C_p3(value=1., mu=.5, tau=1.):
    """C_p3 ~ lognormal(mu, tau)"""
    if value<=0.:
        raise LikelihoodError
    return flib.lognormal(value, mu, tau)


# obs_x and tau are now parents of C.
@node
def C(  eval_fun = Matern, 
        base_mesh = x, 
        obs_mesh = obs_x, 
        obs_tau = tau, 
        diff_degree=C_p1, 
        amp=C_p2, 
        scale=C_p3):
        
    C = Covariance(eval_fun, base_mesh, diff_degree=diff_degree, amp=amp, scale=scale)
    observe_cov(C, obs_mesh, array([obs_tau, obs_tau]))
    return C

# The data d are now a parent of M, and so is C. The data now have no parents.

@data
def d(value=array([3.1, 2.9])):
    """d has no parents"""
    return 0.


# Prior parameters of M

@parameter
def M_p1(value=1., mu=1., tau=1.):
    """M_p1 ~ lognormal(mu, tau)"""
    return normal(value, mu, tau)
    
@parameter
def M_p2(value=.5, mu = .5, tau = 1.):
    """M_p2 ~ normal(mu, tau)"""
    return normal(value, mu, tau)
    
@parameter
def M_p3(value=2., mu=2., tau=1.):
    """M_p3 ~ normal(mu, tau)"""
    return normal(value, mu, tau)


# M itself now has C for a parent
    
def linfun(x, a, b, c):
    return a * x ** 2 + b * x + c
    
@node
def M(  eval_fun = linfun, 
        base_mesh = x, 
        C = C,
        obs_vals = d,
        a=M_p2, 
        b=M_p2, 
        c=M_p3):
        
    M=Mean(eval_fun, base_mesh, a=a, b=a, c=c)
    observe_mean_from_cov(M, C, obs_vals)
    return M


# The Gaussian process parameter f is still valued as a 
# Realization object, but now it has no children.
@parameter
def f(value=None, M=M, C=C):

    def logp(value, M, C):
        return GP_logp(value,M,C)

    def random(M, C):
        return Realization(M,C)

    rseed = 1.

# A GPMetropolis SamplingMethod to handle f    
S = GPMetropolis(f=f, M=M, C=C)
