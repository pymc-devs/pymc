import theano.tensor as tt
import numpy as np
from .mean import Zero

__all__ = ['GPP']

class GPP(object):
    """Gausian process prior
    
    Parameters
    ----------
    mean_func : Mean
        Mean function of Gaussian process
    cov_func : Covariance
        Covariance function of Gaussian process
    """
    def __init__(self, mean_func=None, cov_func=None):
        
        if mean_func is None:
            self.mean_func = Zero()
        else:
            self.mean_func = mean_func
            
        if cov_func is None:
            raise ValueError('A covariance function must be specified for GPP')
        self.cov_func = cov_func
        
    def __call__(self, X):
        """Evaluate GPP at grid of inputs"""
        