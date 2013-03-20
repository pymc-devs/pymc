import numpy as np
from pymc import  *
from pymc.distributions.multivariate import Dirichlet


model = Model()
Var = model.Var
Data = model.Data 

k = 5
a = constant(np.array([2,3.,4, 2,2]))

p, p_m1 = model.TransformedVar(
                 'p', Dirichlet(k,a), 
                 simplextransform, shape = k - 1)


H = model.d2logpc()

s = find_MAP(model)

step = HamiltonianMC(model, model.vars, H(s))
trace, _,t = sample(1000, step, s) 


