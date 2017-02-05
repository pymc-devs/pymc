# Elliptical slice sampler

import numpy as np
import numpy.random as nr
import theano.tensor as tt

from .arraystep import ArrayStep, Competence
from ..model import modelcontext
from ..theanof import inputvars
from ..vartypes import continuous_types
from ..distributions import draw_values

__all__ = ['EllipticalSlice']


class EllipticalSlice(ArrayStep):
    """Multivariate elliptical slice sampler step.

    Parameters
    ----------
    vars : list
        List of variables for sampler.
    prior_cov : array
        Covariance matrix of the multivariate Gaussian prior.
    logp : function
        Log likelihood.
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    """
    default_blocked = False

    def __init__(self, vars=None, prior_cov=None, model=None, **kwargs):
        self.model = modelcontext(model)
        self.prior_cov = prior_cov

        # Won't work if prior_cov is of dimension 1 (univariate normal)
        self.prior_mean = tt.zeros_like(prior_cov.diagonal())

        if vars is None:
            vars = self.model.cont_vars
        vars = inputvars(vars)

        super(EllipticalSlice, self).__init__(vars, [self.model.fastlogp], **kwargs)

    def astep(self, q0, logp):
        """q0 : current state
        logp : log probability function
        """
        nu = nr.multivariate_normal(mean=draw_values([self.prior_mean]),
                                    cov=draw_values([self.prior_cov]))
        y = logp(q0) - nr.standard_exponential()

        theta = nr.uniform(0, 2 * np.pi)
        theta_max = theta
        theta_min = theta - 2 * np.pi

        q_new = q0 * np.cos(theta) + nu * np.sin(theta)

        while logp(q_new) <= y:
            if theta < 0:
                theta_min = theta
            else:
                theta_max = theta
            theta = nr.uniform(theta_min, theta_max)
            q_new = q0 * np.cos(theta) + nu * np.sin(theta)

        return q_new
