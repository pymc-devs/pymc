#   Copyright 2023 The PyMC Developers
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
import pytensor.tensor as pt

from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable

from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _moment,
)
from pymc.distributions.shape_utils import _change_dist_size, change_dist_size
from pymc.util import check_dist_not_registered


class CensoredRV(SymbolicRandomVariable):
    """Censored random variable"""

    inline_logprob = True
    _print_name = ("Censored", "\\operatorname{Censored}")


class Censored(Distribution):
    r"""
    Censored distribution

    The pdf of a censored distribution is

    .. math::

        \begin{cases}
            0 & \text{for } x < lower, \\
            \text{CDF}(lower, dist) & \text{for } x = lower, \\
            \text{PDF}(x, dist) & \text{for } lower < x < upper, \\
            1-\text{CDF}(upper, dist) & \text {for} x = upper, \\
            0 & \text{for } x > upper,
        \end{cases}


    Parameters
    ----------
    dist : unnamed_distribution
        Univariate distribution which will be censored.
        This distribution must have a logcdf method implemented for sampling.

        .. warning:: dist will be cloned, rendering it independent of the one passed as input.

    lower : float or None
        Lower (left) censoring point. If `None` the distribution will not be left censored
    upper : float or None
        Upper (right) censoring point. If `None`, the distribution will not be right censored.

    Warnings
    --------
    Continuous censored distributions should only be used as likelihoods.
    Continuous censored distributions are a form of discrete-continuous mixture
    and as such cannot be sampled properly without a custom step sampler.
    If you wish to sample such a distribution, you can add the latent uncensored
    distribution to the model and then wrap it in a :class:`~pymc.Deterministic`
    :func:`~pymc.math.clip`.


    Examples
    --------
    .. code-block:: python

        with pm.Model():
            normal_dist = pm.Normal.dist(mu=0.0, sigma=1.0)
            censored_normal = pm.Censored("censored_normal", normal_dist, lower=-1, upper=1)
    """

    rv_type = CensoredRV

    @classmethod
    def dist(cls, dist, lower, upper, **kwargs):
        if not isinstance(dist, TensorVariable) or not isinstance(
            dist.owner.op, (RandomVariable, SymbolicRandomVariable)
        ):
            raise ValueError(
                f"Censoring dist must be a distribution created via the `.dist()` API, got {type(dist)}"
            )
        if dist.owner.op.ndim_supp > 0:
            raise NotImplementedError(
                "Censoring of multivariate distributions has not been implemented yet"
            )
        check_dist_not_registered(dist)
        return super().dist([dist, lower, upper], **kwargs)

    @classmethod
    def rv_op(cls, dist, lower=None, upper=None, size=None):
        lower = pt.constant(-np.inf) if lower is None else pt.as_tensor_variable(lower)
        upper = pt.constant(np.inf) if upper is None else pt.as_tensor_variable(upper)

        # When size is not specified, dist may have to be broadcasted according to lower/upper
        dist_shape = size if size is not None else pt.broadcast_shape(dist, lower, upper)
        dist = change_dist_size(dist, dist_shape)

        # Censoring is achieved by clipping the base distribution between lower and upper
        dist_, lower_, upper_ = dist.type(), lower.type(), upper.type()
        censored_rv_ = pt.clip(dist_, lower_, upper_)

        return CensoredRV(
            inputs=[dist_, lower_, upper_],
            outputs=[censored_rv_],
            ndim_supp=0,
        )(dist, lower, upper)


@_change_dist_size.register(CensoredRV)
def change_censored_size(cls, dist, new_size, expand=False):
    uncensored_dist, lower, upper = dist.owner.inputs
    if expand:
        new_size = tuple(new_size) + tuple(uncensored_dist.shape)
    return Censored.rv_op(uncensored_dist, lower, upper, size=new_size)


@_moment.register(CensoredRV)
def moment_censored(op, rv, dist, lower, upper):
    moment = pt.switch(
        pt.eq(lower, -np.inf),
        pt.switch(
            pt.isinf(upper),
            # lower = -inf, upper = inf
            0,
            # lower = -inf, upper = x
            upper - 1,
        ),
        pt.switch(
            pt.eq(upper, np.inf),
            # lower = x, upper = inf
            lower + 1,
            # lower = x, upper = x
            (lower + upper) / 2,
        ),
    )
    moment = pt.full_like(dist, moment)
    return moment
