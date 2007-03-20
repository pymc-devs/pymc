from PyMC2 import Sampler
from PyMC2.examples import DisasterModel
M = Sampler(DisasterModel)


# Sample
M.sample(50000,0,100)
