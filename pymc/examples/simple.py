import pymc

true_mu = 1.5
true_tau = 50.0
N_samples = 500

mu = pymc.Uniform('mu', lower=-100.0, upper=100.0)
tau = pymc.Uniform('tau', lower=0.0, upper=100.0)

data = pymc.rnormal( true_mu, true_tau, size=(N_samples,) )

y = pymc.Normal('y',mu,tau,value=data,observed=True)
