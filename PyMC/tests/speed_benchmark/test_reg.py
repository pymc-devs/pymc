from PyMC import Model
from PyMC.examples import DisasterModel
M = Model(DisasterModel)


# Sample
M.sample(50000,0,100)
