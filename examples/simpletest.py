from mcex import *
import numpy as np 

#import pydevd 
#pydevd.set_pm_excepthook()


data = np.random.normal(size = 3)

x = FreeVariable('x', (3,), 'float64')
x_prior = Normal(value = x, mu = .5, tau = 2.**-2)
ydata = Normal(value = data, mu = x, tau = .75**-2)


chain = ChainState({'x' : [0.2,.3,.1]})



hmc_model = Model([x], 
                  logps = [x_prior, ydata ],
                  derivative_vars = [x])
mapping = VariableMapping([x])

find_map(mapping, hmc_model, chain)
hmc_cov = approx_hessian(mapping, hmc_model, chain)

sampler = Sampler([hmc.HMCStep(hmc_model,mapping, hmc_cov)])


ndraw = 1e6
history = SampleHistory(hmc_model, ndraw)
print ("took :", sample(ndraw, sampler, chain, history))
