"""
Test interactive sampler
"""

# TODO: Make real test case.

from pymc import MCMC
from pymc.examples import DisasterModel

S = MCMC(DisasterModel)
S.interactive_sample(1000,100,2)
