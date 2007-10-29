"""
Test interactive sampler
"""

# TODO: Make real test case.

from PyMC import Sampler
from PyMC.examples import DisasterModel

S = Sampler(DisasterModel)
S.interactive_sample(10000,1000,2)
