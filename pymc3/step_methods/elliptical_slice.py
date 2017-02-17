import numpy as np
import numpy.random as nr
import theano.tensor as tt

from .arraystep import ArrayStep, Competence
from ..model import modelcontext
from ..theanof import inputvars
from ..distributions import draw_values

__all__ = ['EllipticalSlice']


class EllipticalSlice(ArrayStep):
    """Multivariate elliptical slice sampler step.

    Elliptical slice sampling (ESS) [1]_ is a variant of slice sampling
    that allows sampling from distributions with multivariate Gaussian
    prior and arbitrary likelihood. It is generally about as fast as
    regular slice sampling, mixes well even when the prior covariance
    might otherwise induce a strong dependence between samples, and
    does not depend on any tuning parameters.

    The Gaussian prior is assumed to have zero mean.

    Parameters
    ----------
    vars : list
        List of variables for sampler.
    prior_cov : array
        Covariance matrix of the multivariate Gaussian prior.
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from
        context).

    References
    ----------
    .. [1] I. Murray, R. P. Adams, and D. J. C. MacKay. "Elliptical Slice
       Sampling", The Proceedings of the 13th International Conference on
       Artificial Intelligence and Statistics (AISTATS), JMLR W&CP
       9:541-548, 2010.
    """

    default_blocked = True

    def __init__(self, vars=None, prior_cov=None, model=None, **kwargs):
        self.model = modelcontext(model)
        self.prior_cov = tt.as_tensor_variable(prior_cov)
        self.prior_mean = tt.zeros_like(self.prior_cov[0])

        if vars is None:
            vars = self.model.cont_vars
        vars = inputvars(vars)

        super(EllipticalSlice, self).__init__(vars, [self.model.fastlogp], **kwargs)

    def astep(self, q0, logp):
        """q0 : current state
        logp : log probability function
        """
        mean, cov = draw_values([self.prior_mean, self.prior_cov])
        nu = nr.multivariate_normal(mean=mean, cov=cov)
        y = logp(q0) - nr.standard_exponential()

        # Draw initial proposal and propose a candidate point
        theta = nr.uniform(0, 2 * np.pi)
        theta_max = theta
        theta_min = theta - 2 * np.pi
        q_new = q0 * np.cos(theta) + nu * np.sin(theta)

        while logp(q_new) <= y:
            # Shrink the bracket and propose a new point
            if theta < 0:
                theta_min = theta
            else:
                theta_max = theta
            theta = nr.uniform(theta_min, theta_max)
            q_new = q0 * np.cos(theta) + nu * np.sin(theta)

        return q_new

    @staticmethod
    def competence(var):
        # Because it requires a specific type of prior, this step method
        # should only be assigned explicitly.
        return Competence.INCOMPATIBLE
