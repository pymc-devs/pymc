import numpy as np
from pymc import  *

from AR1 import AR1
from scipy.sparse import csc_matrix


returns_d = np.genfromtxt("SP500.csv")


model = Model()
Var = model.Var
Data = model.Data 

n = 400

gam = Var('gam', Normal(-5., 3.**-2))
k = Var('k', Uniform(0,1), testval = .95)

tau, ltau = model.TransformedVar(
    'tau', Bound(T(.05,.05,10), 0), transform = exp, logjacobian = lambda x: x, testval = -2.3)

lvol = Var('lvol', AR1(k, tau**-2), shape = n)

rtau = exp((lvol+gam)*-2.)

lreturns = Data(returns_d[-n:], Normal(0, rtau))



#fitting

start = find_MAP(model, vars = [lvol])
start = find_MAP(model,start, vars = [gam, k])

params = [gam, k, ltau]
H = model.d2logpc()

def hessian(q, tau_scale): 
    h = H(q)
    h[2,2] = abs(h[2,2]) / tau_scale
    h[2,3:] = 0
    h[3:,2] = 0
    return csc_matrix(h)

st = start

step = hmc_step(model, model.vars, hessian(st, .5), step_size_scaling = .05, trajectory_length = 4.)
trace, _,t = sample(1000, step, st) 
st = trace.point(-1)

step = hmc_step(model, model.vars, hessian(st, 2.), step_size_scaling = .5, trajectory_length = 4.)
trace, _,t = sample(10000, step, st) 
