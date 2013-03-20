'''
Created on May 12, 2012

@author: john
'''
from ..core import *
from arraystep import * 
from numpy import array, max, exp, cumsum, nested_iters, empty, searchsorted, ones
from numpy.random import uniform

from theano.gof.graph import inputs
    
__all__ = ['ElemwiseCategoricalStep']
    
class ElemwiseCategoricalStep(ArrayStep):
    """
    Gibbs sampling for categorical variables that only have only have elementwise effects
    the variable can't be indexed into or transposed or anything otherwise that will mess things up
    
    """
    #TODO: It would be great to come up with a way to make ElemwiseCategoricalStep  more general (handling more complex elementwise variables)
    def __init__(self, model, var, values):
        self.sh = ones(var.dshape, var.dtype)
        self.values = values
        self.var = var
        
        ArrayStep.__init__(self, [var], [elemwise_logp(model, var)])
    
    def astep(self, state, q, logp):
        p = array([logp(v * self.sh) for v in self.values])
        return state, categorical(p, self.var.dshape)



def elemwise_logp(model, var):
    terms = [term for term in model.factors       if var in inputs([term])] 
    return add(*terms)

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




