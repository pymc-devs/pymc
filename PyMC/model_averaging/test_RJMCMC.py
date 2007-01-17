from RJMCMC import *
import transd_model
RM = Model(transd_model)


RM.sample(100,10,1)