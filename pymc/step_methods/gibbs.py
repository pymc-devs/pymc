'''
Created on May 12, 2012

@author: john
'''
from ..core import *
from numpy import array, max, exp, cumsum, nested_iters, empty, searchsorted, ones
from numpy.random import uniform
from __builtin__ import sum as builtin_sum
    
    
class elemwise_cat_gibbs_step():
    """
    gibbs sampling for categorical variables that only have only have elementwise effects
    the variable can't be indexed into or transposed or anything otherwise that will mess things up
    
    It would be great to come up with a way to make this more general (handling more complex elementwise variables)
    """
    def __init__(self, model, var, values):
        self.elogp = elemwise_logp(model, var)
        self.sh = ones(var.dshape, var.dtype)
        self.values = values
        self.var = var
        
    
    def step(self, state, point):
        bij = DictElemBij(self.var, (), point)
        logp = bij.mapf(self.elogp)

        p = array([logp(v * self.sh) for v in self.values])
        return state, bij.rmap(categorical(p, self.var.dshape))

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

from theano.gof.graph import inputs





def elemwise_logp(model, var):
    terms = filter(lambda term: var in inputs([term]), model.factors)

    p = function(model.vars, builtin_sum(terms))
    return KWArgFunc(p)
