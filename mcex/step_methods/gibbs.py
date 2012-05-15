'''
Created on May 12, 2012

@author: john
'''
from ..core import *
from numpy import max, exp, cumsum, nested_iters, empty, searchsorted
from numpy.random import uniform
    
    
def categorical_gibbs_step(var, conditionalp):
    
    def step( state, chain):
        chain = chain.copy()    
        chain[str(var)] = categorical(conditionalp(**chain), var.dshape)
        return state, chain
    
    return step



def categorical(prob, shape) :
    out = empty([1] + list(shape))
    
    n = len(shape)
    it0, it1 = nested_iters([prob, out], [range(1,n +1),[0]] ,
                            op_flags = [['readonly'], ['readwrite']],
                            flags = ['reduce_ok'])
    
    for i in it0: 
        p, o = it1.itviews
        p = cumsum(exp(p - max(p, axis =0)))
        r = uniform() * p[-1]
        
        o[0] = searchsorted(p, r)
        
    return out[0,...]