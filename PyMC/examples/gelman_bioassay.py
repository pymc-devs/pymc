from PyMC2 import *
from numpy import *

n = 5*ones(4,dtype=int)
dose=array([-.86,-.3,-.05,.73])

@stoch
def alpha(value=-1., mu=0, tau=.00001):
    """alpha ~ normal(mu, V)"""
    return 0.*normal_like(value, mu, tau)

@stoch
def beta(value=10., mu=0, tau=.00001):
    """beta ~ normal(mu, V)"""
    return 0.*normal_like(value, mu, tau)

@dtrm
def theta(a=alpha, b=beta, d=dose):
    """theta = inv_logit(a+b)"""
    return invlogit(a+b*d)

@data
@stoch
def deaths(value=array([0,1,3,5]), n=n, p=theta):
    """deaths ~ binomial(n, p)"""
    return binomial_like(value, n, p)
