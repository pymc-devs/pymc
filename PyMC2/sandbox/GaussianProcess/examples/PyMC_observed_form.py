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
def C_diff_degree(value=1.4, mu=1.4, tau=1.):
    """C_diff_degree ~ lognormal(mu, tau)"""
    if value<=0.:
        raise LikelihoodError
    return flib.lognormal(value, mu, tau)
    
@parameter
def C_amp(value=.4, mu = .4, tau = 1.):
    """C_amp ~ lognormal(mu, tau)"""
    if value<=0.:
        raise LikelihoodError
    return flib.lognormal(value, mu, tau)
    
@parameter
def C_scale(value=1., mu=.5, tau=1.):
    """C_scale ~ lognormal(mu, tau)"""
    if value<=0.:
        raise LikelihoodError
    return flib.lognormal(value, mu, tau)


# obs_x and tau are now parents of C.
@node
def C(  eval_fun = Matern, 
        base_mesh = x, 
        obs_mesh = obs_x, 
        obs_tau = tau, 
        diff_degree=C_diff_degree, 
        amp=C_amp, 
        scale=C_scale):
 
    val = Covariance(eval_fun, base_mesh, diff_degree=diff_degree, amp=amp, scale=scale)
    observe_cov(val, obs_mesh, obs_taus = array([obs_tau, obs_tau]))
    return val

# The data d are now a parent of M, and so is C. The data now have no parents.

@data
def d(value=array([3.1, 2.9])):
    """d has no parents"""
    return 0.


# Prior parameters of M

@parameter
def M_a(value=1., mu=1., tau=1.):
    """M_a ~ lognormal(mu, tau)"""
    return flib.normal(value, mu, tau)
    
@parameter
def M_b(value=.5, mu = .5, tau = 1.):
    """M_b ~ normal(mu, tau)"""
    return flib.normal(value, mu, tau)
    
@parameter
def M_c(value=2., mu=2., tau=1.):
    """M_c ~ normal(mu, tau)"""
    return flib.normal(value, mu, tau)


# M itself now has C for a parent
    
def linfun(x, a, b, c):
    return a * x ** 2 + b * x + c
    
@node
def M(  eval_fun = linfun, 
        base_mesh = x, 
        C = C,
        obs_vals = d,
        a=M_a, 
        b=M_b, 
        c=M_c):

    val=Mean(eval_fun, base_mesh, a=a, b=b, c=c)
    observe_mean_from_cov(val, C, obs_vals)
    return val


# The Gaussian process parameter f is still valued as a 
# Realization object, but now it has no children.
@parameter
def f(value=Realization(M,C), M=M, C=C):

    def logp(value, M, C):
        return GP_logp(value,M,C)

    def random(M, C):
        return Realization(M,C)
        

# A GPMetropolis SamplingMethod to handle f    
S = GPMetropolis(f=f, M=M, C=C)
