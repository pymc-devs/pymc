import scipy.stats.distributions as sc_dst
import inspect
import numpy as np
from pymc import Stochastic
from copy import copy
from distributions import *

"""
Wraps a SciPy rv object in a PyMC random variable object. 
Needs to wrap the following methods:

generic.rvs(<shape(s)>,loc=0,scale=1)
    - random variates

generic.pdf(x,<shape(s)>,loc=0,scale=1)
    - probability density function

generic.cdf(x,<shape(s)>,loc=0,scale=1)
    - cumulative density function

generic.sf(x,<shape(s)>,loc=0,scale=1)
    - survival function (1-cdf --- sometimes more accurate)

generic.ppf(q,<shape(s)>,loc=0,scale=1)
    - percent point function (inverse of cdf --- percentiles)

generic.isf(q,<shape(s)>,loc=0,scale=1)
    - inverse survival function (inverse of sf)

generic.stats(<shape(s)>,loc=0,scale=1,moments='mv')
    - mean('m',axis=0), variance('v'), skew('s'), and/or kurtosis('k')

generic.entropy(<shape(s)>,loc=0,scale=1)
    - (differential) entropy of the RV.
"""
__all__ = ['stochastic_from_scipy_dist']
def separate_shape_args(kwds, shape_args):

    new_kwds = {}
    for name in kwds:
        new_kwds[name] = kwds[name]

    args = [new_kwds.pop(name) for name in shape_args]

    return args, new_kwds

def stochastic_from_scipy_dist(scipy_dist):
    """
    Return a Stochastic subclass made from a particular SciPy distribution.
    """
    
    name = scipy_dist.__class__.__name__.replace('_gen','').capitalize()


    (args, varargs, varkw, defaults) = inspect.getargspec(scipy_dist._cdf)
    shape_args = args[2:]            
    if isinstance(scipy_dist, sc_dst.rv_continuous):
        dtype=float
    
        def logp(value, **kwds):
            args, kwds = separate_shape_args(kwds, shape_args)
            return np.sum(np.log(scipy_dist.pdf(value,*args,**kwds)))

        parent_names = shape_args + ['loc', 'scale']
        defaults = [None] * (len(parent_names)-2) + [0., 1.]
        
    elif isinstance(scipy_dist, sc_dst.rv_discrete):
        dtype=int

        def logp(value, **kwds):
            args, kwds = separate_shape_args(kwds, shape_args)
            return np.sum(np.log(scipy_dist.pmf(value,*args,**kwds)))

        parent_names = shape_args + ['loc']
        defaults = [None] * (len(parent_names)-1) + [0]        
    else:
        return None
    

    parents_default = dict(zip(parent_names, defaults))

        
    def random(shape=None,**kwds):
        args, kwds = separate_shape_args(kwds, shape_args)

        if shape is None:
            return scipy_dist.rvs(*args,**kwds)
        else:
            return np.reshape(scipy_dist.rvs(*args,**kwds), shape)
    
    # Build docstring from distribution
    docstr = name[0]+' = '+name + '(name, '+', '.join(parent_names)+', value=None, shape=None, trace=True, rseed=True, doc=None)\n\n'
    docstr += 'Stochastic variable with '+name+' distribution.\nParents are: '+', '.join(parent_names) + '.\n\n'
    docstr += """
Methods:

    random()
        - draws random value
          sets value to return value

    ppf(q)
        - percent point function (inverse of cdf --- percentiles)
          sets value to return value

    isf(q)
        - inverse survival function (inverse of sf)
          sets value to return value

    stats(moments='mv')
        - mean('m',axis=0), variance('v'), skew('s'), and/or kurtosis('k')


Attributes:

    logp
        - sum(log(pdf())) or sum(log(pmf()))

    cdf
        - cumulative distribution function

    sf
        - survival function (1-cdf --- sometimes more accurate)

    entropy
        - (differential) entropy of the RV.

        
NOTE: If you encounter difficulties with this object, please try the analogous 
computation using the rv objects in scipy.stats.distributions directly before 
reporting the bug.
    """
        
    new_class = new_dist_class(dtype, name, parent_names, parents_default, docstr, logp, random)
    class newer_class(new_class):
        __doc__ = docstr
        rv = scipy_dist
        def __init__(self, *args, **kwds):
            new_class.__init__(self, *args, **kwds)
            self.args, self.kwds = separate_shape_args(self.parents, shape_args)        
            self.frozen_rv = self.rv(self.args, self.kwds)

        def _cdf(self):
            """
            The cumulative distribution function of self conditional on parents 
            evaluated at self's current value
            """
            return self.rv.cdf(self.value, *self.args, **self.kwds)
        cdf = property(_cdf, doc=_cdf.__doc__)
    
        def _sf(self):
            """
            The survival function of self conditional on parents
            evaluated at self's current value
            """
            return self.rv.sf(self.value, *self.args, **self.kwds)
        sf = property(_sf, doc=_sf.__doc__)
    
        def ppf(self, q):
            """
            The percentile point function (inverse cdf) of self conditional on parents.
            Self's value will be set to the return value.
            """
            self.value = self.rv.ppf(q, *self.args, **self.kwds)
            return self.value
        
        def isf(self, q):
            """
            The inverse survival function of self conditional on parents.
            Self's value will be set to the return value.
            """
            self.value = self.rv.isf(q, *self.args, **self.kwds)       
            return self.value
        
        def stats(self, moments='mv'):
            """The first few moments of self's distribution conditional on parents"""
            return self.rv.stats(moments=moments, *self.args, **self.kwds)
        
        def _entropy(self):
            """The entropy of self's distribution conditional on its parents"""
            return self.rv.entropy(*self.args, **self.kwds)
        entropy = property(_entropy, doc=_entropy.__doc__)

    newer_class.__name__ = new_class.__name__
    return newer_class
    
for scipy_dist_name in sc_dst.__all__:
    scipy_dist = sc_dst.__dict__[scipy_dist_name]
    if isinstance(scipy_dist, sc_dst.rv_continuous) or isinstance(scipy_dist, sc_dst.rv_discrete):
        new_dist = stochastic_from_scipy_dist(scipy_dist)
        locals()[new_dist.__name__] = new_dist
        __all__.append(new_dist.__name__)
