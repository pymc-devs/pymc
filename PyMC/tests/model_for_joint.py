from proposition5 import *

# For some reason the decorator isn't working with no parents...
@parameter(init_val = ones(3,dtype='float'))
def A():
    def logp_fun(value):
        return sum(-1.*value**2)
        
@parameter(init_val = ones(5,dtype='float'))
def B():
    def logp_fun(value):
        return sum(-1.*value**2)        
        
S = Joint([A,B])
