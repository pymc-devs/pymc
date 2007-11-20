"""
Test interactive sampler
"""

# TODO: Make real test case.

from PyMC import MCMCSampler
from PyMC.examples import DisasterModel

S = MCMCSampler(DisasterModel)
S.interactive_sample(10000,1000,2)
