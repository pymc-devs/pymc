from abc import ABCMeta
import numbers
from copy import copy

import statsmodels.api as sm

from .links import *
import pymc

__all__ = ['Normal', 'T', 'Binomial', 'Poisson']

class Family(object):
    __metaclass__ = ABCMeta
    priors = {}
    link = Identity

    def __init__(self, **kwargs):
        # Overwrite defaults
        for key, val in kwargs.iteritems():
            if key == 'priors':
                self.priors = copy(self.priors)
                self.priors.update(val)
            else:
                setattr(self, key, val)

        # Instantiate link function
        self.link_func = self.link()

    def get_priors(self, model=None):
        model = pymc.modelcontext(model)
        priors = {}
        for key, val in self.priors.iteritems():
            if isinstance(val, numbers.Number):
                priors[key] = val
            else:
                priors[key] = model.Var(val[0], val[1])

        return priors

    def make_model(self, y_est, y_data, model=None):
        priors = self.get_priors(model=model)
        priors[self.parent] = self.link_func.theano(y_est)
        return self.likelihood('y', observed=y_data, **priors)

    def sm_family(self):
        return self.sm_family(self.link.sm())

    def __repr__(self):
        return "{0} Family({1})".format(self.__class__, self.__dict__)


class Normal(Family):
    sm_family = sm.families.Gaussian
    link = Identity
    likelihood = pymc.Normal
    parent = 'mu'
    priors = {'sd': ('sigma', pymc.Uniform.dist(0, 100))}

class T(Family):
    sm_family = sm.families.Gaussian
    link = Identity
    likelihood = pymc.T
    parent = 'mu'
    priors = {'lam': ('sigma', pymc.Uniform.dist(0, 100)),
              'nu': 1}


class Binomial(Family):
    link = Logit
    sm_family = sm.families.Binomial
    likelihood = pymc.Bernoulli
    parent = 'p'

class Poisson(Family):
    link = Log
    sm_family = sm.families.Poisson
    likelihood = pymc.Poisson
    parent = ''
