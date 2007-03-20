"""
Test interactive sampler
"""

from PyMC2 import Sampler
from PyMC2.examples import DisasterModel

S = Sampler(DisasterModel)
S.interactive_sample(10000,1000,2, verbose=False)
