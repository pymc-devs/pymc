import pymc
from pymc import six
from . import simple_von_mises

model=pymc.MCMC(simple_von_mises)
model.sample(iter=1000, burn=500, thin=2)

six.print_('mu',model.mu.value)
six.print_('kappa',model.kappa.value)
