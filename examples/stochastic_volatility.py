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



sd, lsd = model.TransformedVar(
                 'sd', Exponential(1./.02),
                 transforms.log, testval = -2.5)

nu = Var('nu', Exponential(1./10))

lvol = Var('lvol', timeseries.RW(sd**-2), shape = n)

lreturns = Data(returns, T(nu, lam = exp(-2*lvol)))



#fitting

H = model.d2logpc()

def hessian(q, nusd): 
    h = H(q)
    h[1,1] = nusd**-2
    h[:2,2:] = h[2:,:2] = 0

    return csc_matrix(h)


from  scipy import optimize
s = find_MAP(model, vars = [lvol], fmin = optimize.fmin_l_bfgs_b)
s = find_MAP(model, s, vars = [lsd, nu])

step = hmc_step(model, model.vars, hessian(s, 6))
trace, _,t = sample(200, step, s) 

s2 = trace.point(-1)

step = hmc_step(model, model.vars, hessian(s2, 6), trajectory_length = 4.)
trace, _,t = sample(4000, step, trace = trace) 

