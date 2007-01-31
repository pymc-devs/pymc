from PyMC2 import Model
from PyMC2.examples import DisasterModel
M = Model(DisasterModel)


# Sample
M.sample(5000,0,100)
