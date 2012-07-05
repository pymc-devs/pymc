'''
Created on Jul 5, 2012

@author: jsalvatier
'''
import numpy as np

def make_univariate(var, idx, C, f):
    def univariate(x):
        c = C.copy()
        v = c[var].copy()
        v[idx] = x 
        c[var] = v
        return f(c)
    return univariate


    
    
def hist_covar(hist, vars):
    def flat_h(var):
        x = hist[str(var)]
        return x.reshape((x.shape[0], np.prod(x.shape[1:])))
    
    return np.cov(np.concatenate(map(flat_h, vars), 1).T)