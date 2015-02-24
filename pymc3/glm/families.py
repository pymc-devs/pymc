import numbers
from copy import copy

try:
    import statsmodels.genmod.families.family as sm_family
except ImportError:
    GaussianSM = None
    BinomialSM = None
    PoissonSM = None

from .links import *
from ..model import modelcontext
import ..distributions as pm_dists

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
    sm_family = sm_family.Gaussian
    link = Identity
    likelihood = pm_dists.Normal
    parent = 'mu'
    priors = {'sd': pm_dists.HalfCauchy.dist(0, 100)}


class T(Family):
    sm_family = sm_family.Gaussian
    link = Identity
    likelihood = pm_dists.T
    parent = 'mu'
    priors = {'lam': pm_dists.HalfCauchy.dist(0, 100)),
              'nu': 1}


class Binomial(Family):
    link = Logit
    sm_family = sm_family.Binomial
    likelihood = pm_dists.Bernoulli
    parent = 'p'


class Poisson(Family):
    link = Log
    sm_family = sm_family.Poisson
    likelihood = pm_dists.Poisson
    parent = 'mu'
    priors = {'mu': pm_dists.HalfNormal.dist(sd=1)}
