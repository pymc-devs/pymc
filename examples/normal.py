from pymc import *
import numpy as np 
import theano 

#import pydevd 
#pydevd.set_pm_excepthook()
np.seterr(invalid = 'raise')

data = np.random.normal(size = (3, 20))
n = 1

model = Model()
with model: 
    x = Normal('x', 0, 1., n)

    #start sampling at the MAP
    start = find_MAP()
    h = approx_hess(start) #find a good orientation using the hessian at the MAP

    step = HamiltonianMC(model.vars, h, is_cov = False)

    ndraw = 3e3
    trace = sample(ndraw, step, start)

