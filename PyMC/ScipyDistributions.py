import scipy.stats.distributions as sc_dst
import inspect
import numpy as np
from PyMC import Stochastic, DiscreteStochastic
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

def separate_shape_args(kwds, shape_args):

    new_kwds = {}
    for name in kwds:
        new_kwds[name] = kwds[name]

    args = [new_kwds.pop(name) for name in shape_args]

    return args, new_kwds

def stoch_from_scipy_dist(scipy_dist):
    """
    Return a Stochastic subclass made from a particular scipy distribution.
    """
    
    name = 'Scipy'+scipy_dist.__class__.__name__.replace('_gen','').capitalize()


    (args, varargs, varkw, defaults) = inspect.getargspec(scipy_dist._cdf)
    shape_args = args[2:]            
    if isinstance(scipy_dist, sc_dst.rv_continuous):
        base = Stochastic
    
        def logp(value, **kwds):
            args, kwds = separate_shape_args(kwds, shape_args)
            return np.sum(np.log(scipy_dist.pdf(value,*args,**kwds)))

        parent_names = shape_args + ['loc', 'scale']
        defaults = [None] * (len(parent_names)-2) + [0., 1.]
        
    elif isinstance(scipy_dist, sc_dst.rv_discrete):
        base = DiscreteStochastic

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
        - draws random values and returns them

    logp()
        - sum(log(pdf())) or sum(log(pmf()))

    cdf()
        - cumulative distribution function

    sf()
        - survival function (1-cdf --- sometimes more accurate)

    ppf(q)
        - percent point function (inverse of cdf --- percentiles)
          sets value to return value

    isf(q)
        - inverse survival function (inverse of sf)
          sets value to return value

    stats(moments='mv')
        - mean('m',axis=0), variance('v'), skew('s'), and/or kurtosis('k')

    entropy()
        - (differential) entropy of the RV.
    """
        
    new_class = new_dist_class(base, name, parent_names, parents_default, docstr, logp, random)
    
    def cdf_wrap(self):
        args, kwds = separate_shape_args(self.parents, shape_args)
        return scipy_dist.cdf(self.value, *args, **kwds)
    
    def sf_wrap(self):
        args, kwds = separate_shape_args(self.parents, shape_args)
        return scipy_dist.sf(self.value, *args, **kwds)
    
    def ppf_wrap(self, q):
        args, kwds = separate_shape_args(self.parents, shape_args)
        self.value = scipy_dist.ppf(q, *args, **kwds)
        return self.value
        
    def isf_wrap(self, q):
        args, kwds = separate_shape_args(self.parents, shape_args)
        self.value = scipy_dist.isf(q, *args, **kwds)
        return self.value
        
    def stats_wrap(self, moments='mv'):
        args, kwds = separate_shape_args(self.parents, shape_args)        
        return scipy_dist.stats(moments=moments, *args, **kwds)
        
    def entropy_wrap(self):
        args, kwds = separate_shape_args(self.parents, shape_args)        
        return scipy_dist.entropy(*args, **kwds)

    for method_name in ['cdf_wrap', 'sf_wrap', 'entropy_wrap']:
        setattr(new_class, method_name[:-5], property(locals()[method_name]))

    for method_name in ['ppf_wrap', 'isf_wrap', 'stats_wrap']:
        setattr(new_class, method_name[:-5], locals()[method_name])

    return new_class
    
for scipy_dist_name in sc_dst.__all__:
    scipy_dist = sc_dst.__dict__[scipy_dist_name]
    if isinstance(scipy_dist, sc_dst.rv_continuous) or isinstance(scipy_dist, sc_dst.rv_discrete):
        new_dist = stoch_from_scipy_dist(scipy_dist)
        locals()[new_dist.__name__] = new_dist
