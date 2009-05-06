from PyMCObjects import Stochastic
from Container import Container
from InstantiationDecorators import stochastic
from flib import mod_to_circle
from distributions import rvon_mises, von_mises_like
import numpy as np
from distributions import valuewrapper

__all__ = ['CircularStochastic', 'CircVonMises']

class CircularStochastic(Stochastic):
    """
    C = CircularStochastic(lo, hi, *args, **kwargs)

    Takes two special parents, lo and hi; any incoming value
    will be mapped into the interval [lo,hi).

    args and kwargs will be passed to Stochastic.__init__.

    :SeeAlso: Stochastic
    """
    def __init__(self, lo, hi, *args, **kwargs):
        self.interval_parents = Container([hi, lo])
        Stochastic.__init__(self, *args, **kwargs)

    def set_value(self, value):
        modded_value = mod_to_circle(value, *self.interval_parents.value).reshape(np.shape(value))
        Stochastic.set_value(self, modded_value)
    value = property(Stochastic.get_value, set_value)

class CircVonMises(CircularStochastic):
    """
    V = CircVonMises(name, mu, kappa, value=None, observed=False, size=1, trace=True, rseed=True, doc=None, verbose=0)

    Stochastic variable with Von Mises distribution.
    Parents are: mu, kappa.

    Docstring of log-probability function:
    """+von_mises_like.__doc__
    def __init__(   self,
                    name,
                    mu, kappa,
                    value=None,
                    observed=False,
                    size=1,
                    trace=True,
                    cache_depth=2,
                    rseed=True,
                    plot=None,
                    verbose = 0):

        if value is None:
            arg_eval = Container([mu, kappa]).value
            value = rvon_mises(arg_eval[0], arg_eval[1], size=size)

        parents = {'mu':mu, 'kappa':kappa}
        logp = valuewrapper(von_mises_like)
        random = lambda mu, kappa, size=size: rvon_mises(mu, kappa, size=size)
        CircularStochastic.__init__(self, -np.pi, np.pi, logp, 'A Von Mises-distributed variable', name, parents, random, trace, value, np.dtype('float'), rseed, observed, cache_depth, plot, verbose)
