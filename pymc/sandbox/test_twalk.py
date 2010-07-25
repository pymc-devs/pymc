from pymc.sandbox.TWalk import *
from pymc import *
from numpy import random, inf
import pdb

"""
Test model for T-walk algorithm:

Suppose x_{i,j} ~ Be( theta_j ), i=0,1,2,...,n_j-1, ind. j=0,1,2
But it is known that 0 <  theta_0 < theta_3 < theta_2 < 1
"""

theta_true = array([ 0.4, 0.5, 0.7 ])  ### True thetas
n = array([ 20, 15, 40]) ### sample sizes

#### Simulated data, but we only need the sum of 1's
r = zeros(3)
for j in range(3):
	r[j] = sum(random.random(size=n[j]) < theta_true[j])

@stochastic
def theta(value=(0.45, 0.5, 0.55)):
    """theta ~ beta(alpha, beta)"""
    a,b,c = value
    if not a<b<c:
        return -inf
    return uniform_like(value, 0, 1)

# Binomial likelihood
x = Binomial('x', n=n, p=theta, value=r, observed=True)

# Using standard Metropolis sampling
M = MCMC([theta, x])
M.use_step_method(TWalk, theta, inits=(0.3,0.4,0.5), verbose=1)
M.sample(50000, 40000, verbose=2)
Matplot.plot(M)