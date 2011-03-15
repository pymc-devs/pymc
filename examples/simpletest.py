from mcex import *
import numpy as np 

data = np.random.normal(size = 3)

x = dvector('x')
x_prior = Normal(value = x, mu = .5, tau = 2.**-2)
ydata = Normal(value = data, mu = x, tau = .75**-2)


model = Model()

chain = ChainState({'x' : [0.2,.3,.1]})



hmc_model = model.submodel([x], 
                          logps = [x_prior, ydata ],
                          gradient_vars = [x])
mapping = VariableMapping([x])

find_map(mapping, hmc_model, chain)
hmc_cov = approx_hess(mapping, hmc_model, chain)

sampler = Sampler([hmc.HMCStep(hmc_model, cov)])


sample(500, sampler, chain)
