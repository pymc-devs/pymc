'''
Created on Jul 5, 2012

@author: jsalvatier
'''
import numpy as np

def make_univariate(var, idx, C, f):
    """
    convert a function that takes a parameter point into one that takes a single value
    for a specific parameter holding all the other parameters constant.
    """
    def univariate(x):
        c = C.copy()
        v = c[var].copy()
        v[idx] = x 
        c[var] = v
        return f(c)
    return univariate


    
    
def hist_covar(hist, vars):
    """calculate the flattened covariance matrix using a sample history"""
    def flat_h(var):
        x = hist[str(var)]
        return x.reshape((x.shape[0], np.prod(x.shape[1:])))
    
    return np.cov(np.concatenate(map(flat_h, vars), 1).T)