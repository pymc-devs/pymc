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
import theano

from theano import tensor as tt

import pymc3 as pm

from pymc3.variational import opvi
from pymc3.variational.opvi import ObjectiveFunction, Operator
from pymc3.variational.stein import Stein

__all__ = ["KL", "KSD"]


class KL(Operator):
    R"""**Operator based on Kullback Leibler Divergence**

    This operator constructs Evidence Lower Bound (ELBO) objective

    .. math::

        ELBO_\beta = \log p(D|\theta) - \beta KL(q||p)

    where

    .. math::

        KL[q(v)||p(v)] = \int q(v)\log\frac{q(v)}{p(v)}dv


    Parameters
    ----------
    approx: :class:`Approximation`
        Approximation used for inference
    beta: float
        Beta parameter for KL divergence, scales the regularization term.
    """

    def __init__(self, approx, beta=1.0):
        Operator.__init__(self, approx)
        self.beta = pm.floatX(beta)

    def apply(self, f):
        return -self.datalogp_norm + self.beta * (self.logq_norm - self.varlogp_norm)


# SVGD Implementation


class KSDObjective(ObjectiveFunction):
    R"""Helper class for construction loss and updates for variational inference

    Parameters
    ----------
    op: :class:`KSD`
        OPVI Functional operator
    tf: :class:`TestFunction`
        OPVI TestFunction
    """

    def __init__(self, op, tf):
        if not isinstance(op, KSD):
            raise opvi.ParametrizationError("Op should be KSD")
        ObjectiveFunction.__init__(self, op, tf)

    @theano.config.change_flags(compute_test_value="off")
    def __call__(self, nmc, **kwargs):
        op = self.op  # type: KSD
        grad = op.apply(self.tf)
        if self.approx.all_histograms:
            z = self.approx.joint_histogram
        else:
            z = self.approx.symbolic_random
        if "more_obj_params" in kwargs:
            params = self.obj_params + kwargs["more_obj_params"]
        else:
            params = self.test_params + kwargs["more_tf_params"]
            grad *= pm.floatX(-1)
        grads = tt.grad(None, params, known_grads={z: grad})
        return self.approx.set_size_and_deterministic(
            grads, nmc, 0, kwargs.get("more_replacements")
        )


class KSD(Operator):
    R"""**Operator based on Kernelized Stein Discrepancy**

    Input: A target distribution with density function :math:`p(x)`
        and a set of initial particles :math:`\{x^0_i\}^n_{i=1}`

    Output: A set of particles :math:`\{x_i\}^n_{i=1}` that approximates the target distribution.

    .. math::

        x_i^{l+1} \leftarrow \epsilon_l \hat{\phi}^{*}(x_i^l) \\
        \hat{\phi}^{*}(x) = \frac{1}{n}\sum^{n}_{j=1}[k(x^l_j,x) \nabla_{x^l_j} logp(x^l_j)/temp +
        \nabla_{x^l_j} k(x^l_j,x)]

    Parameters
    ----------
    approx: :class:`Approximation`
        Approximation used for inference
    temperature: float
        Temperature for Stein gradient

    References
    ----------
    -   Qiang Liu, Dilin Wang (2016)
        Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm
        arXiv:1608.04471
    """
    has_test_function = True
    returns_loss = False
    require_logq = False
    objective_class = KSDObjective

    def __init__(self, approx, temperature=1):
        Operator.__init__(self, approx)
        self.temperature = temperature

    def apply(self, f):
        # f: kernel function for KSD f(histogram) -> (k(x,.), \nabla_x k(x,.))
        stein = Stein(
            approx=self.approx,
            kernel=f,
            use_histogram=self.approx.all_histograms,
            temperature=self.temperature,
        )
        return pm.floatX(-1) * stein.grad
