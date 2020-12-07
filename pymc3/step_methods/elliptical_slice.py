#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import numpy.random as nr
import theano.tensor as tt

from pymc3.distributions import draw_values
from pymc3.model import modelcontext
from pymc3.step_methods.arraystep import ArrayStep, Competence
from pymc3.theanof import inputvars

__all__ = ["EllipticalSlice"]


def get_chol(cov, chol):
    """Get Cholesky decomposition of the prior covariance.

    Ensure that exactly one of the prior covariance or Cholesky
    decomposition is passed. If the prior covariance was passed, then
    return its Cholesky decomposition.

    Parameters
    ----------
    cov: array, optional
        Covariance matrix of the multivariate Gaussian prior.
    chol: array, optional
        Cholesky decomposition of the covariance matrix of the
        multivariate Gaussian prior.
    """

    if len([i for i in [cov, chol] if i is not None]) != 1:
        raise ValueError("Must pass exactly one of cov or chol")

    if cov is not None:
        chol = tt.slinalg.cholesky(cov)
    return chol


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
    vars: list
        List of variables for sampler.
    prior_cov: array, optional
        Covariance matrix of the multivariate Gaussian prior.
    prior_chol: array, optional
        Cholesky decomposition of the covariance matrix of the
        multivariate Gaussian prior.
    model: PyMC Model
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

    def __init__(self, vars=None, prior_cov=None, prior_chol=None, model=None, **kwargs):
        self.model = modelcontext(model)
        chol = get_chol(prior_cov, prior_chol)
        self.prior_chol = tt.as_tensor_variable(chol)

        if vars is None:
            vars = self.model.cont_vars
        vars = inputvars(vars)

        super().__init__(vars, [self.model.fastlogp], **kwargs)

    def astep(self, q0, logp):
        """q0: current state
        logp: log probability function
        """

        # Draw from the normal prior by multiplying the Cholesky decomposition
        # of the covariance with draws from a standard normal
        chol = draw_values([self.prior_chol])[0]
        nu = np.dot(chol, nr.randn(chol.shape[0]))
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
    def competence(var, has_grad):
        # Because it requires a specific type of prior, this step method
        # should only be assigned explicitly.
        return Competence.INCOMPATIBLE
