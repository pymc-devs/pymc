import numpy as np
from pymc import  *
from pymc.distributions.multivariate import Dirichlet


model = Model()
with model:

    k = 5
    a = constant(np.array([2,3.,4, 2,2]))

    p, p_m1 = model.TransformedVar(
                 'p', Dirichlet(k,a, shape = k), 
                 simplextransform)


H = model.d2logpc()

s = find_MAP(model)

step = HamiltonianMC(model, model.vars, H(s))
trace, _,t = sample(1000, step, s) 


