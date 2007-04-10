# GaussianProcess/examples/PyMC_unobserved_form.py

from GaussianProcess import *
from GaussianProcess.cov_funs import Matern
from numpy import *
from pylab import *
from PyMC2 import *
from PyMC2 import flib


x = arange(-1.,1.,.1)
obs_x = array([-.5,.5])

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

# The covariance node C is valued as a Covariance object.                    
@node
def C(eval_fun = Matern, base_mesh = x, diff_degree=C_p1, amp=C_p2, scale=C_p3):
    return Covariance(eval_fun, base_mesh, diff_degree=diff_degree, amp=amp, scale=scale)


# Prior parameters of M

@parameter
def M_p1(value=1., mu=1., tau=1.):
    """M_p1 ~ lognormal(mu, tau)"""
    return flib.normal(value, mu, tau)

@parameter
def M_p2(value=.5, mu = .5, tau = 1.):
    """M_p2 ~ normal(mu, tau)"""
    return flib.normal(value, mu, tau)

@parameter
def M_p3(value=2., mu=2., tau=1.):
    """M_p3 ~ normal(mu, tau)"""
    return flib.normal(value, mu, tau)

# The mean node M is valued as a Mean object.
def linfun(x, a, b, c):
    return a * x ** 2 + b * x + c
    
@node
def M(eval_fun = linfun, base_mesh = x, a=M_p1, b=M_p2, c=M_p3):
    return Mean(eval_fun, base_mesh, a=a, b=b, c=c)


# The Gaussian process parameter f is valued as a Realization object.
@parameter
def f(value=None, M=M, C=C):

    def logp(value, M, C):
        return GP_logp(value,M,C)

    def random(M, C):
        return Realization(M,C)

    rseed = 1.

# A GPMetropolis SamplingMethod to handle f    
S = GPMetropolis(f=f, M=M, C=C)


# Observation precision
@parameter
def tau(value=500., alpha = 1., beta = 500.):
    """tau ~ gamma(alpha, beta)"""
    if value<=0.:
        raise LikelihoodError
    return flib.gamma(value, alpha, beta)

# The data d is just array-valued. It's normally distributed about f(obs_x).
@data
def d(value=array([3.1, 2.9]), x=obs_x, mu=f, tau=tau):
    """d ~ N(f(obs_x), tau)"""
    mu_array = mu(obs_x)
    return flib.normal(value, mu_array, tau)