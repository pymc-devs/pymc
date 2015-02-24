import numbers
from copy import copy

import theano.tensor
from ..model import modelcontext
import ..distributions as pm_dists

__all__ = ['Normal', 'T', 'Binomial', 'Poisson']

# Define link functions
identity = lambda self, x: x
logit = theano.tensor.nnet.sigmoid
inverse = theano.tensor.inv
log = theano.tensor.log

class Family(object):
    """Base class for Family of likelihood distribution and link functions.
    """
    priors = {}
    link = identity

    def __init__(self, **kwargs):
        # Overwrite defaults
        for key, val in kwargs.items():
            if key == 'priors':
                self.priors = copy(self.priors)
                self.priors.update(val)
            else:
                setattr(self, key, val)

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
                priors[key] = model.Var(key, val)

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
        priors[self.parent] = self.link(y_est)
        return self.likelihood('y', observed=y_data, **priors)

    def __repr__(self):
        return """Family {klass}:
    Likelihood   : {likelihood}({parent})
    Priors       : {priors}
    Link function: {link}.""".format(klass=self.__class__, likelihood=self.likelihood.__name__, parent=self.parent, priors=self.priors, link=self.link)


class Normal(Family):
    link = identity
    likelihood = pm_dists.Normal
    parent = 'mu'
    priors = {'sd': pm_dists.HalfCauchy.dist(beta=10)}


class T(Family):
    link = identity
    likelihood = pm_dists.T
    parent = 'mu'
    priors = {'lam': pm_dists.HalfCauchy.dist(beta=10)),
              'nu': 1}


class Binomial(Family):
    link = logit
    likelihood = pm_dists.Bernoulli
    parent = 'p'


class Poisson(Family):
    link = log
    likelihood = pm_dists.Poisson
    parent = 'mu'
    priors = {'mu': pm_dists.HalfCauchy.dist(beta=10)}
