import numpy as np
from pymc import  *

from scipy.sparse import csc_matrix

#data
returns = np.genfromtxt("data/SP500.csv")

n = 400
returns = returns[-n:]

#model

model = Model()
Var = model.Var
Data = model.Data 



gam = Var('gam', Normal(-5., 3.**-2))
k = Var('k', Uniform(0,1), testval = .95)

tau, ltau = model.TransformedVar(
                 'tau', Tpos(10,.05,.05),
                 transform = exp, logjacobian = lambda x: x, testval = -2.3)

lvol = Var('lvol', timeseries.AR1(k, tau**-2), shape = n)

rtau = exp((lvol+gam)*-2.)

lreturns = Data(returns, Normal(0, rtau))



#fitting

start = find_MAP(model, vars = [lvol])
start = find_MAP(model,start, vars = [gam, k])

H = model.d2logpc()

def hessian(q, tau_scale): 
    h = H(q)

    h[2,2] = abs(h[2,2]) / tau_scale
    h[2,3:] = h[3:,2] = 0

    return csc_matrix(h)


step = hmc_step(model, model.vars, hessian(start, .5), step_size_scaling = .05, trajectory_length = 4.)
trace, _,t = sample(1000, step, start) 
start = trace.point(-1)

#step = hmc_step(model, model.vars, hessian(start, 2.), step_size_scaling = .5, trajectory_length = 4.)
#trace, _,t = sample(10000, step, start) 
