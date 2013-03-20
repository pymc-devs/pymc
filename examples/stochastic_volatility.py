import numpy as np
from pymc import  *
from pymc.distributions.timeseries import *

from scipy.sparse import csc_matrix
from  scipy import optimize

"""
1. Data
-------
"""
returns = np.genfromtxt("data/SP500.csv")

n = 400
returns = returns[-n:]

"""
2. Build Model
--------------
Stochastic volatility model described in Hoffman (2011) on p21.
"""
model = Model()
Var = model.Var
Data = model.Data 

#its easier to sample the scale of the volatility process innovations on a log scale 
sd, log_sd = model.TransformedVar('sd', Exponential(1./.02),
                 logtransform, testval = -2.5)

nu = Var('nu', Exponential(1./10))

lvol = Var('lvol', GaussianRandomWalk(sd**-2), shape = n)

lreturns = Data(returns, T(nu, lam = exp(-2*lvol)))

"""
3. Fit Model
------------
"""
H = model.d2logpc()

"""
To get a decent scale for the hamiltonaian sampler, we find the hessian at a point. However, the 2nd derivatives for the degrees of freedom are negative and thus not very informative, so we make an educated guess. The interactions between lsd/nu and lvol are also not very useful, so we set them to zero. 

The hessian matrix is also very sparse, so we make it a sparse matrix for faster sampling.
"""
def hessian(point, nusd): 
    h = H(point)
    h[1,1] = nusd**-2
    h[:2,2:] = h[2:,:2] = 0

    return csc_matrix(h)

"""
the full MAP is a degenerate case wrt to sd and nu, so we find the MAP wrt the volatility process, keeping log_sd and nu constant at their default values. we use l_bfgs_b because it is more efficient for high dimensional functions (lvol has n elements)
"""

s = find_MAP(model, vars = [lvol], fmin = optimize.fmin_l_bfgs_b)

#we do a short initial run to get near the right area
step = hmc_step(model, model.vars, hessian(s, 6))
trace, _,t = sample(200, step, s) 

#then start again using a new hessian at the new start
s2 = trace.point(-1)
step = hmc_step(model, model.vars, hessian(s2, 6), trajectory_length = 4.)
trace, _,t = sample(4000, step, trace = trace) 

"""
4. References
-------------
    1. Hoffman & Gelman. (2011). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. http://arxiv.org/abs/1111.4246 
"""
