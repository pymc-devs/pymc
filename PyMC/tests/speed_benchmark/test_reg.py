from PyMC import Sampler
from PyMC.examples import DisasterModel
M = Sampler(DisasterModel)


# Sample
M.sample(50000,0,100)
