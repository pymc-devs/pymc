from PyMC2 import Model
import fastDisasterModel
M = Model(fastDisasterModel)


# Sample
M.sample(50000,0,100)
