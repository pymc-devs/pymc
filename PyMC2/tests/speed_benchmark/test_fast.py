from PyMC2 import Model
import fastDisasterModel
M = Model(fastDisasterModel)


# Sample
M.sample(5000,0,10)
