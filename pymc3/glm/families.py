import numbers
from copy import copy

try:
    from statsmodels.genmod.families.family import (Gaussian, Binomial, Poisson)
except ImportError:
    Gaussian = None
    Binomial = None
    Poisson = None

from .links import *
from ..model import modelcontext
from ..distributions import Normal, T, Uniform, Bernoulli, Poisson

__all__ = ['Normal', 'T', 'Binomial', 'Poisson']

class Family(object):
    """Base class for Family of likelihood distribution and link functions.
    """
    priors = {}
    link = Identity

    def __init__(self, **kwargs):
        # Overwrite defaults
        for key, val in kwargs.items():
            if key == 'priors':
                self.priors = copy(self.priors)
                self.priors.update(val)
            else:
                setattr(self, key, val)

        # Instantiate link function
        self.link_func = self.link()

    def _get_priors(self, model=None):
        """Return prior distributions of the likelihood.

        Returns
        -------
        dict : mapping name -> pymc3 distribution
        """
        model = modelcontext(model)
        priors = {}
        for key, val in self.priors.items():
            if isinstance(val, numbers.Number):
                priors[key] = val
            else:
                priors[key] = model.Var(val[0], val[1])

        return priors

    def create_likelihood(self, y_est, y_data, model=None):
        """Create likelihood distribution of observed data.

        Parameters
        ----------
        y_est : theano.tensor
            Estimate of dependent variable
        y_data : array
            Observed dependent variable
        """
        priors = self._get_priors(model=model)
        # Wrap y_est in link function
        priors[self.parent] = self.link_func.theano(y_est)
        return self.likelihood('y', observed=y_data, **priors)

    def create_statsmodel_family(self):
        """Instantiate and return statsmodel family object.
        """
        if self.sm_family is None:
            return None
        else:
            return self.sm_family(self.link.sm)

    def __repr__(self):
        return """Family {klass}:
    Likelihood   : {likelihood}({parent})
    Priors       : {priors}
    Link function: {link}.""".format(klass=self.__class__, likelihood=self.likelihood.__name__, parent=self.parent, priors=self.priors, link=self.link)


class Normal(Family):
    sm_family = Gaussian
    link = Identity
    likelihood = Normal
    parent = 'mu'
    priors = {'sd': ('sigma', Uniform.dist(0, 100))}


class T(Family):
    sm_family = Gaussian
    link = Identity
    likelihood = T
    parent = 'mu'
    priors = {'lam': ('sigma', Uniform.dist(0, 100)),
              'nu': 1}


class Binomial(Family):
    link = Logit
    sm_family = Binomial
    likelihood = Bernoulli
    parent = 'p'


class Poisson(Family):
    link = Log
    sm_family = Poisson
    likelihood = Poisson
    parent = ''
