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
import aesara.tensor as at
import numpy as np

from aesara.scalar import Clip
from aesara.tensor import TensorVariable
from aesara.tensor.random.op import RandomVariable

from pymc.distributions.distribution import SymbolicDistribution, _moment
from pymc.util import check_dist_not_registered


class Censored(SymbolicDistribution):
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
    dist: PyMC unnamed distribution
        PyMC distribution created via the `.dist()` API, which will be censored. This
        distribution must be univariate and have a logcdf method implemented.
    lower: float or None
        Lower (left) censoring point. If `None` the distribution will not be left censored
    upper: float or None
        Upper (right) censoring point. If `None`, the distribution will not be right censored.


    Examples
    --------
    .. code-block:: python

        with pm.Model():
            normal_dist = pm.Normal.dist(mu=0.0, sigma=1.0)
            censored_normal = pm.Censored("censored_normal", normal_dist, lower=-1, upper=1)
    """

    @classmethod
    def dist(cls, dist, lower, upper, **kwargs):
        if not isinstance(dist, TensorVariable) or not isinstance(dist.owner.op, RandomVariable):
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
    def rv_op(cls, dist, lower=None, upper=None, size=None, rngs=None):
        if lower is None:
            lower = at.constant(-np.inf)
        if upper is None:
            upper = at.constant(np.inf)

        # Censoring is achieved by clipping the base distribution between lower and upper
        rv_out = at.clip(dist, lower, upper)

        # Reference nodes to facilitate identification in other classmethods, without
        # worring about possible dimshuffles
        rv_out.tag.dist = dist
        rv_out.tag.lower = lower
        rv_out.tag.upper = upper

        if size is not None:
            rv_out = cls.change_size(rv_out, size)
        if rngs is not None:
            rv_out = cls.change_rngs(rv_out, rngs)

        return rv_out

    @classmethod
    def ndim_supp(cls, *dist_params):
        return 0

    @classmethod
    def change_size(cls, rv, new_size, expand=False):
        dist_node = rv.tag.dist.owner
        lower = rv.tag.lower
        upper = rv.tag.upper
        rng, old_size, dtype, *dist_params = dist_node.inputs
        new_size = new_size if not expand else tuple(new_size) + tuple(old_size)
        new_dist = dist_node.op.make_node(rng, new_size, dtype, *dist_params).default_output()
        return cls.rv_op(new_dist, lower, upper)

    @classmethod
    def change_rngs(cls, rv, new_rngs):
        (new_rng,) = new_rngs
        dist_node = rv.tag.dist.owner
        lower = rv.tag.lower
        upper = rv.tag.upper
        olg_rng, size, dtype, *dist_params = dist_node.inputs
        new_dist = dist_node.op.make_node(new_rng, size, dtype, *dist_params).default_output()
        return cls.rv_op(new_dist, lower, upper)

    @classmethod
    def graph_rvs(cls, rv):
        return (rv.tag.dist,)


@_moment.register(Clip)
def moment_censored(op, rv, dist, lower, upper):
    moment = at.switch(
        at.eq(lower, -np.inf),
        at.switch(
            at.isinf(upper),
            # lower = -inf, upper = inf
            0,
            # lower = -inf, upper = x
            upper - 1,
        ),
        at.switch(
            at.eq(upper, np.inf),
            # lower = x, upper = inf
            lower + 1,
            # lower = x, upper = x
            (lower + upper) / 2,
        ),
    )
    moment = at.full_like(dist, moment)
    return moment
