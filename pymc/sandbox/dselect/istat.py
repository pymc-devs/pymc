"""Distribution parameters given the mean and variance."""

import numpy as np
from scipy import special as sp
from scipy.stats import skew, kurtosis
import defaults
import pymc

def beta(**kwds):
    """Return the parameters of the beta distribution (alpha, beta) given the
    mean and variance."""
    mean = kwds['mean']
    var = kwds['variance']
    
    alpha = -(mean*var + mean**3 - mean**2) / var
    beta = ((mean-1)*var + mean**3 - 2*mean**2 + mean) / var
    return alpha, beta
    
def cauchy(**kwds):
    """Return the parameters of the cauchy distribution (alpha, beta) given the
    median and entropy."""
    alpha = kwds['median']
    E = kwds['entropy']
    beta = np.exp(E)/4./np.pi
    return alpha, beta
    
def chi2(**kwds):
    """Return the parameter of the chi2 distribution (nu) given the mean."""
    return kwds['mean']
    
def exponential(**kwds):
    """Return the parameter of the exponential distribution (beta) given the
    mean."""
    return 1./kwds['mean']
    
def exponweib(**kwds):
    """Return the parameters of the exponweib distribution (alpha, k, loc, 
    scale) given the mean and variance."""
    # Here we need to optimize based on the theoretica moments
    raise NotImplementedError


def gamma(**kwds):
    """Return the parameters of the gamma distribution (alpha, beta)
    given the mean and variance."""
    mean = kwds['mean']
    var = kwds['variance']
    beta = mean/var
    alpha = mean * beta
    return alpha, beta

def weibull(**kwds):
    """
    median=\beta \ln(2)^{1/\alpha}
    """
    #mean = kwds['mean']
    #var = kwds['variance']
    median = kwds['median']
    mode = kwds['mode']
    raise NotImplementedError
    
        
def entropy(r):
    """Compute a naive estimator of the Shannon entropy."""
    p, b = pymc.utils.histogram(r, 'scott', normed=True)
    p[p==0] += p.mean()*1e-10
    return -(p * np.log(p)).sum()
    
def mode(r):
    """Return an approximative mode."""
    p, b = pymc.utils.histogram(r, 'scott', normed=True)
    i = p.argmax()
    return b['bincenters'][i]
    
def describe(r):
    """Return a dictionary with various statistics computed on r:
        mean, variance, skew, kurtosis, entropy, median.
    """
    stats = {}
    stats['mean'] = r.mean()
    stats['variance'] = r.var()
    stats['skew'] = skew(r)
    stats['kurtosis'] = kurtosis(r)
    stats['median'] = np.median(r)
    stats['entropy'] = entropy(r)
    stats['mode'] = mode(r)
    return stats
