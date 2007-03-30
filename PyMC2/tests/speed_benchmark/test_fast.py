from PyMC2 import Sampler
import fastDisasterModel
M = Sampler(fastDisasterModel)


# Sample
M.sample(5000,0,10)
