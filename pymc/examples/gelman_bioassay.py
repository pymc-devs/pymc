from pymc import *
from numpy import ones, array

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

"""deaths ~ binomial(n, p)"""    
deaths = Binomial('deaths', n=n, p=theta, value=array([0,1,3,5], dtype=float), observed=True)
