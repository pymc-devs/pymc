__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

import pymc
import numpy as np
try:
    from GaussianSubmodel import GaussianSubmodel, slice_by_stochastics

class NormApproxMu(object):
    """
    Returns the mean vector of some variables.
    
    Usage: If p1 and p2 are array-valued stochastic variables and N is a 
    NormApprox or MAP object,
    
    N.mu(p1,p2)
    
    will give the approximate posterior mean of the ravelled, concatenated
    values of p1 and p2.
    """
    def __init__(self, owner):
        self.owner = owner
    
    def __getitem__(self, *stochastics):
        
        if not owner.fitted:
            raise ValueError, 'NormApprox object must be fitted before mu can be accessed.'
        
        tot_len = 0
        
        try:
            for p in stochastics[0]:
                pass
            stochastic_tuple = stochastics[0]
        except:
            stochastic_tuple = stochastics
        
        for p in stochastic_tuple:
            tot_len += self.owner.stochastic_len[p]
            
        mu = zeros(tot_len, dtype=float)
        
        start_index = 0
        for p in stochastic_tuple:
            mu[start_index:(start_index + self.owner.stochastic_len[p])] = self.owner._mu[self.owner._slices[p]]
            start_index += self.owner.stochastic_len[p]
            
        return mu
        

class NormApproxC(object):
    """
    Returns the covariance matrix of some variables.
    
    Usage: If p1 and p2 are array-valued stochastic variables and N is a
    NormApprox or MAP object,
    
    N.C(p1,p2)
    
    will give the approximate covariance matrix of the ravelled, concatenated 
    values of p1 and p2
    """
    def __init__(self, owner):
        self.owner = owner
            
    def __getitem__(self, *stochastics):
        
        if not owner.fitted:
            raise ValueError, 'NormApprox object must be fitted before C can be accessed.'
        
        tot_len = 0
        
        try:
            for p in stochastics[0]:
                pass
            stochastic_tuple = stochastics[0]
        except:
            stochastic_tuple = stochastics
        
        for p in stochastic_tuple:
            tot_len += self.owner.stochastic_len[p]

        C = asmatrix(zeros((tot_len, tot_len)), dtype=float)
            
        start_index1 = 0
        for p1 in stochastic_tuple:
            start_index2 = 0
            for p2 in stochastic_tuple:                
                C[start_index1:(start_index1 + self.owner.stochastic_len[p1]), \
                start_index2:(start_index2 + self.owner.stochastic_len[p2])] = \
                self.owner._C[self.owner._slices[p1],self.owner._slices[p2]]
                
                start_index2 += self.owner.stochastic_len[p2]
                
            start_index1 += self.owner.stochastic_len[p1]
            
        return C