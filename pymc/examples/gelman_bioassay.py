from pymc import *
from numpy import *

n = 5*ones(4,dtype=int)
dose=array([-.86,-.3,-.05,.73])

@stochastic
def alpha(value=-1.):
    return 0.

@stochastic
def beta(value=10.):
    return 0.

@deterministic
def theta(a=alpha, b=beta, d=dose):
    """theta = inv_logit(a+b)"""
    return invlogit(a+b*d)

@data
@stochastic(dtype=int)
def deaths(value=array([0,1,3,5],dtype=float), n=n, p=theta):
    """deaths ~ binomial(n, p)"""
    return binomial_like(value, n, p)