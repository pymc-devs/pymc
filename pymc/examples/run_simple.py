import pymc
from pymc import six
from . import simple
from numpy import mean

model=pymc.MCMC(simple)
model.sample(iter=1000, burn=500, thin=2)

print_('mu',mean(model.trace('mu')[:]))
print_('tau',mean(model.trace('tau')[:]))
