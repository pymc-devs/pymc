"""Utility functions for PyMC"""


# License: Scipy compatible
# Author: David Huard, 2006
# A bit slow. try to optimize
import numpy as np
import sys, inspect
try:
    from scipy import special, factorial
    from scipy import comb
except ImportError:
    print 'Warning, SciPy special functions not available'
from copy import copy
from PyMCObjects import Parameter, Node, PyMCBase
from numpy.linalg.linalg import LinAlgError
from numpy.linalg import cholesky, eigh
from numpy import sqrt

#
# Find PyMC object's random children.
#
def extend_children(pymc_object):
    new_children = copy(pymc_object.children)
    need_recursion = False
    node_children = set()
    for child in pymc_object.children:
        if isinstance(child,Node):
            new_children |= child.children
            node_children.add(child)
            need_recursion = True
    pymc_object.children = new_children - node_children
    if need_recursion:
        extend_children(pymc_object)
    return

#
# Returns a matrix square root of a covariance matrix.
#
def msqrt(cov):
    # Try Cholesky factorization
    try:
        sig = cholesky(cov)
    
    # If there's a small eigenvalue, diagonalize
    except LinAlgError:
        val, vec = eigh(cov)
        sig = np.zeros(vec.shape)
        for i in range(len(val)):
            if val[i]<0.:
                val[i]=0.
            sig[:,i] = vec[:,i]*sqrt(val[i])
    return np.asmatrix(sig).T

def _push(seq,new_value):
    """
    Usage:
    _push(seq,new_value)

    Put a deep copy of new_value at the beginning of seq, and kick out the last value.
    """
    length = len(seq)
    for i in range(length-1):
        seq[i+1] = seq[i]
    if isinstance(seq,ndarray):
        # ndarrays will automatically make a copy
        seq[0] = new_value
    else:
        seq[0] = copy(new_value)



def _extract(__func__, kwds, keys): 
    """
    Used by decorators parameter and node to inspect declarations
    """
    kwds.update({'doc':__func__.__doc__, 'name':__func__.__name__})
    parents = {}

    def probeFunc(frame, event, arg):
        if event == 'return':
            locals = frame.f_locals
            kwds.update(dict((k,locals.get(k)) for k in keys))
            sys.settrace(None)
        return probeFunc

    # Get the __func__tions logp and random (complete interface).
    sys.settrace(probeFunc)
    try:
        __func__()
    except:
        if 'logp' in keys:  
            kwds['logp']=__func__
        else:
            kwds['eval'] =__func__

    for key in keys:
        if not kwds.has_key(key):
            kwds[key] = None            
            
    for key in ['logp', 'eval']:
        if key in keys:
            if kwds[key] is None:
                kwds[key] = __func__

    # Build parents dictionary by parsing the __func__tion's arguments.
    (args, varargs, varkw, defaults) = inspect.getargspec(__func__)
    try:
        parents.update(dict(zip(args[-len(defaults):], defaults)))

    # No parents at all     
    except TypeError: 
        pass
        
    if parents.has_key('value'):
        value = parents.pop('value')
    else:
        value = None
        
    return (value, parents)


"""Exceptions"""

class LikelihoodError(ValueError):
    "Log-likelihood is invalid or negative informationnite"
    pass


def histogram(a, bins=10, range=None, normed=False, weights=None, axis=None):
    """histogram(a, bins=10, range=None, normed=False, weights=None, axis=None)
                                                                   -> H, dict

    Return the distribution of sample.

    Parameters
    ----------
    a:       Array sample.
    bins:    Number of bins, or
             an array of bin edges, in which case the range is not used.
    range:   Lower and upper bin edges, default: [min, max].
    normed:  Boolean, if False, return the number of samples in each bin,
             if True, return the density.
    weights: Sample weights. The weights are normed only if normed is True.
             Should weights.sum() not equal len(a), the total bin count will
             not be equal to the number of samples.
    axis:    Specifies the dimension along which the histogram is computed.
             Defaults to None, which aggregates the entire sample array.

    Output
    ------
    H:            The number of samples in each bin.
                  If normed is True, H is a frequency distribution.
    dict{
    'edges':      The bin edges, including the rightmost edge.
    'upper':      Upper outliers.
    'lower':      Lower outliers.
    'bincenters': Center of bins.
    }

    Examples
    --------
    x = random.rand(100,10)
    H, Dict = histogram(x, bins=10, range=[0,1], normed=True)
    H2, Dict = histogram(x, bins=10, range=[0,1], normed=True, axis=0)

    See also: histogramnd
    """

    a = np.asarray(a)
    if axis is None:
        a = np.atleast_1d(a.ravel())
        axis = 0

    # Bin edges.
    if not np.iterable(bins):
        if range is None:
            range = (a.min(), a.max())
        mn, mx = [mi+0.0 for mi in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        edges = np.linspace(mn, mx, bins+1, endpoint=True)
    else:
        edges = np.asarray(bins, float)

    dedges = np.diff(edges)
    decimal = int(-np.log10(dedges.min())+6)
    bincenters = edges[:-1] + dedges/2.

    # apply_along_axis accepts only one array input, but we need to pass the
    # weights along with the sample. The strategy here is to concatenate the
    # weights array along axis, so the passed array contains [sample, weights].
    # The array is then split back in  __hist1d.
    if weights is not None:
        aw = np.concatenate((a, weights), axis)
        weighted = True
    else:
        aw = a
        weighted = False

    count = np.apply_along_axis(hist1d, axis, aw, edges, decimal, weighted, normed)

    # Outlier count
    upper = count.take(np.array([-1]), axis)
    lower = count.take(np.array([0]), axis)

    # Non-outlier count
    core = a.ndim*[slice(None)]
    core[axis] = slice(1, -1)
    hist = count[core]

    if normed:
        normalize = lambda x: np.atleast_1d(x/(x*dedges).sum())
        hist = np.apply_along_axis(normalize, axis, hist)

    return hist, {'edges':edges, 'lower':lower, 'upper':upper, \
        'bincenters':bincenters}


def hist1d(aw, edges, decimal, weighted, normed):
    """Internal routine to compute the 1d histogram.
    aw: sample, [weights]
    edges: bin edges
    decimal: approximation to put values lying on the rightmost edge in the last
             bin.
    weighted: Means that the weights are appended to array a.
    Return the bin count or frequency if normed.
    """
    nbin = edges.shape[0]+1
    if weighted:
        count = np.zeros(nbin, dtype=float)
        a,w = np.hsplit(aw,2)
        if normed:
            w = w/w.mean()
    else:
        a = aw
        count = np.zeros(nbin, dtype=int)
        w = None


    binindex = np.digitize(a, edges)

    # Values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    on_edge = np.where(np.around(a,decimal) == np.around(edges[-1], decimal))[0]
    binindex[on_edge] -= 1

    # Count the number of identical indices.
    flatcount = np.bincount(binindex, w)

    # Place the count in the histogram array.
    i = np.arange(len(flatcount))
    count[i] = flatcount

    return count




# Some python densities for comparison
def cauchy(x, x0, gamma):
    return 1/pi * gamma/((x-x0)**2 + gamma**2)

def gamma(x, alpha, beta):
    return x**(alpha-1) * exp(-x/beta)/(special.gamma(alpha) * beta**alpha)

def multinomial_beta(alpha):
    nom = (special.gamma(alpha)).prod(0)
    den = special.gamma(alpha.sum(0))
    return nom/den

def dirichlet(x, theta):
    """Dirichlet multivariate probability density.

    :Parameters:
      x : (n,k) array
        Input data
      theta : (n,k) or (1,k) array
        Distribution parameter
    """
    x = np.atleast_2d(x)
    theta = np.atleast_2d(theta)
    f = (x**(theta-1)).prod(0)
    return f/multinomial_beta(theta)

def geometric(x, p):
    return p*(1.-p)**(x-1)

def hypergeometric(x, d, S, N):
    return comb(N-S, x) * comb(S, d-x) / comb(N,d)

def multinomial(x,n,p):
    x = np.atleast_2d(x)
    return factorial(n)/factorial(x).prod(1)*(p**x).prod(1)


