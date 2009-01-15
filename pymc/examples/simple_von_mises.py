import pymc
import numpy as np

true_mu = -0.1
true_kappa = 50.0
N_samples = 500

mu = pymc.Uniform('mu', lower=-np.pi, upper=np.pi)
kappa = pymc.Uniform('kappa', lower=0.0, upper=100.0)


data = pymc.rvon_mises( true_mu, true_kappa, size=(N_samples,) )
y = pymc.VonMises('y',mu, kappa, value=data, observed=True)
