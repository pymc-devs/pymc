#from https://github.com/pymc-devs/pymc/wiki/LatentOccupancy

# Import statements
from numpy import random, array
from pymc import MCMC, Matplot, Beta, Bernoulli, Lambda, Poisson, Uniform, deterministic, logp_of_set, logp_gradient_of_set

n = 100000
theta = 2
pi = 0.4
y = [(random.random()<pi) * random.poisson(theta) for i in range(n)]

def remcache(s):
    s._cache_depth = 0
    s.gen_lazy_function()

p = Beta('p', 1, 1)

z = Bernoulli('z', p, value=array(y)>0, plot=False)

theta_hat = Uniform('theta_hat', 0, 100, value=3)


t = z*theta
counts = Poisson('counts', t, value=y, observed=True)
model = [p, z, theta_hat, counts]

#disable caching for all the nodes
v = model + [t]
for s in v:
    remcache(s)
    
def pymc_logp():
    return logp_of_set(model)

def pymc_dlogp():
    return logp_gradient_of_set(model)