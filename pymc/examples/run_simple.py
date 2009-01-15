import pymc
import simple
from numpy import mean

model=pymc.MCMC(simple)
model.sample(iter=1000, burn=500, thin=2)

print 'mu',mean(model.trace('mu')[:])
print 'tau',mean(model.trace('tau')[:])
