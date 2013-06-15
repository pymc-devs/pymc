import statsmodels.api as sm

from .links import *
import pymc

from abc import ABCMeta

__all__ = ['Normal', 'T', 'Binomial']

class Family(object):
    __metaclass__ = ABCMeta

    def __init__(self, link=None, sm_family=None, likelihood_dist=None, **kwargs):
        if link is not None:
            self.link = link
        if sm_family is not None:
            self.sm_family_cls = sm_family
        if likelihood_dist is not None:
            self.likelihood_dist = likelihood_dist

        self.kwargs = kwargs

        # Instantiate link function
        self.link_func = self.link()

    def likelihood(self, y_est, y_data, model=None):
        return self.likelihood_dist('y', self.link_func.theano(y_est), observed=y_data)

    def sm_family(self):
        return self.sm_family(self.link.sm())

class Normal(Family):
    sm_family = sm.families.Gaussian
    link = Identity

    def likelihood(self, y_est, y_data, model=None):
        model = pymc.modelcontext(model)
        sigma_dist = self.kwargs.get('sigma_dist',  pymc.Uniform.dist(0, 100))
        sigma = model.Var('sigma', sigma_dist)
        y_obs = pymc.Normal('y', mu=self.link_func.theano(y_est), sd=sigma, observed=y_data)
        return y_obs

class T(Family):
    sm_family = sm.families.Gaussian
    link = Identity

    def likelihood(self, y_est, y_data, model=None):
        model = pymc.modelcontext(model)
        sigma_dist = self.kwargs.get('sigma_dist',  pymc.Uniform.dist(0, 100))
        sigma = model.Var('sigma', sigma_dist)
        #nu_dist = self.kwargs.get('sigma_dist',  1)
        #nu = model.Var('nu', nu_dist)

        y_obs = pymc.T('y', mu=self.link_func.theano(y_est), lam=sigma**-2, nu=1, observed=y_data)
        return y_obs

class Binomial(Family):
    link = Logit
    sm_family = sm.families.Binomial
    likelihood_dist = pymc.Bernoulli
