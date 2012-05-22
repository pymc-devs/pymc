'''
Created on May 12, 2012

@author: john
'''
from ..core import *
from numpy import array, max, exp, cumsum, nested_iters, empty, searchsorted, ones
from numpy.random import uniform
    
    
def elemwise_cat_gibbs_step(model, var, values):
    """
    gibbs sampling for categorical variables that only have only have elementwise effects
    the variable can't be indexed into or transposed or anything otherwise that will mess things up
    
    It would be great to come up with a way to make this more general (handling more complex elementwise variables)
    """
    elogp = elemwise_logp(model, var)
    sh = ones(var.dshape, var.dtype);
    
    def step( state, chain):
        def project(x):
            c = chain.copy()
            c[str(var)] = x
            return c 
            
         
        p = array([elogp(project(v * sh)) for v in values])
        return state, project(categorical(p, var.dshape))
    
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

from theano.gof.graph import inputs





def elemwise_logp(model, var):
    terms = filter(lambda term: var in inputs([term]), model.factors)

    p = function(model.vars, builtin_sum(terms))
    def fn(x):
        return p(**x)
    return fn