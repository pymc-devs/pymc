# GaussianProcess/examples/PyMC_unobserved_form.py

from GaussianProcess import *
from GaussianProcess.cov_funs import matern
from numpy import *
from pylab import *
from PyMC2 import *
from PyMC2 import flib


x = arange(-1.,1.,.1)
obs_x = array([-.5,.5])

# Prior parameters of C
@parameter
def C_diff_degree(value=1.4, mu=1.4, tau=1.):
    """C_diff_degree ~ lognormal(mu, tau)"""
    if value<=0.:
        raise ZeroProbability
    return flib.lognormal(value, mu, tau)
    
@parameter
def C_amp(value=.4, mu = .4, tau = 1.):
    """C_amp ~ lognormal(mu, tau)"""
    if value<=0.:
        raise ZeroProbability
    return flib.lognormal(value, mu, tau)
    
@parameter
def C_scale(value=1., mu=.5, tau=1.):
    """C_scale ~ lognormal(mu, tau)"""
    if value<=0.:
        raise ZeroProbability
    return flib.lognormal(value, mu, tau)

# The covariance node C is valued as a Covariance object.                    
@node
def C(eval_fun = matern, base_mesh = x, diff_degree=C_diff_degree, amp=C_amp, scale=C_scale):
    return Covariance(eval_fun, base_mesh, diff_degree=diff_degree, amp=amp, scale=scale)


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

# The mean node M is valued as a Mean object.
def linfun(x, a, b, c):
    return a * x ** 2 + b * x + c
    
@node
def M(eval_fun = linfun, base_mesh = x, a=M_a, b=M_b, c=M_c):
    return Mean(eval_fun, base_mesh, a=a, b=b, c=c)


# This GaussianProcess object

f = GaussianProcess(M=M, C=C, name='f')

# is equivalent to the following PyMC parameter declaration:
#
# @parameter
# def f(value=Realization(M.value,C.value), M=M, C=C):
# 
#     def logp(value, M, C):
#         return GP_logp(value,M,C)
# 
#     def random(M, C):
#         return Realization(M,C)
#
# The only difference is that GaussianProcess objects are automatically handled by
# the GPMetropolis sampling method.


# Observation precision
@parameter
def tau(value=500., alpha = 1., beta = 500.):
    """tau ~ gamma(alpha, beta)"""
    if value<=0.:
        raise ZeroProbability
    return flib.gamma(value, alpha, beta)

# The data d is just array-valued. It's normally distributed about f(obs_x).
@data
def d(value=array([3.1, 2.9]), x=obs_x, mu=f, tau=tau):
    """
    d ~ N(f(obs_x), tau)

    Note that because of f's base mesh:
        f[5] = f(-.5)
        f[-5] = f(.5),
    so
        [f[5], f[-5]] = f(obs_x).
        
    The array aspect is therefore used for value access,
    because it's much faster.
    """
    mu_array = array([mu[5],mu[-5]])
    return flib.normal(value, mu_array, tau)

    
# Uncomment this if you want to use the GPGibbs sampling method instead of the default
# GPMetropolis.
S = ObservedGPGibbs(f=f, M=M, C=C, obs_mesh=obs_x, obs_taus=tau, obs_vals=d)