import numpy as np

# TODO I could not locate this function used anywhere in the code base
# do we need it?
def make_univariate(var, idx, C, f):
    """
    Convert a function that takes a parameter point into one that takes 
    a single value for a specific parameter holding all the other parameters 
    constant.
    
    Parameters
    ----------
    
    """
    def univariate(x):
        c = C.copy()
        v = c[var].copy()
        v[idx] = x 
        c[var] = v
        return f(c)
    return univariate
    
def hist_covar(hist, vars):
    """Calculate the flattened covariance matrix using a sample history"""
    def flat_h(var):
        x = hist[str(var)]
        return x.reshape((x.shape[0], np.prod(x.shape[1:])))
    
    return np.cov(np.concatenate(map(flat_h, vars), 1).T)

# TODO Also could not find this function used anywhere. Are any of the 
# functions in misc necessary?
def autocorr(x):
    x = np.squeeze(x)
    import pylab
    return pylab.acorr( x - np.mean(x))