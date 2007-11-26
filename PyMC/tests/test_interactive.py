"""
Test interactive sampler
"""

# TODO: Make real test case.

from PyMC import MCMC
from PyMC.examples import DisasterModel

S = MCMC(DisasterModel)
S.interactive_sample(10000,1000,2)
