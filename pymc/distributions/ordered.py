#   Copyright 2024 The PyMC Developers
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
import pytensor.tensor as pt

from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import normalize_size_param
from pytensor.tensor.variable import TensorVariable

from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _support_point,
)
from pymc.distributions.shape_utils import change_dist_size, get_support_shape_1d, rv_size_is_none
from pymc.distributions.transforms import _default_transform, ordered


class OrderedRV(SymbolicRandomVariable):
    inline_logprob = True
    extended_signature = "(x)->(x)"
    _print_name = ("Ordered", "\\operatorname{Ordered}")

    @classmethod
    def rv_op(cls, dist, *, size=None):
        # We don't allow passing `rng` because we don't fully control the rng of the components!

        size = normalize_size_param(size)

        if not rv_size_is_none(size):
            core_shape = tuple(dist.shape)[-1]
            shape = (*tuple(size), core_shape)
            dist = change_dist_size(dist, shape)

        sorted_rv = pt.sort(dist, axis=-1)

        return OrderedRV(
            inputs=[dist],
            outputs=[sorted_rv],
        )(dist)


class Ordered(Distribution):
    r"""Univariate IID Ordered distribution.

    The pdf of the oredered distribution is

    .. math::
        f(x_1, ..., x_n) = n!\prod_{i=1}^n f(x_{(i)}),
        where x_1 <= x2 <= ... <= x_n

    Parameters
    ----------
    dist: unnamed_distribution
        Univariate IID distribution which will be sorted.

        .. warning:: dist will be cloned, rendering it independent of the one passade as input

    Examples
    --------
    .. code-block:: python
        import pymc as pm

        with pm.Model():
            x = pm.Normal.dist(mu=0, sigma=1)  # Must be IID
            ordered_x = pm.Ordered("ordered_x", dist=x, shape=(3,))

        pm.draw(ordered_x, random_seed=52)  # array([0.05172346, 0.43970706, 0.91500416])
    """

    rv_type = OrderedRV
    rv_op = OrderedRV.rv_op

    def __new__(cls, name, dist, *, support_shape=None, **kwargs):
        support_shape = get_support_shape_1d(
            support_shape=support_shape,
            shape=None,  # shape will be checked in `cls.dist`
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
        )
        return super().__new__(cls, name, dist, support_shape=support_shape, **kwargs)

    @classmethod
    def dist(cls, dist, *, support_shape=None, **kwargs):
        if not isinstance(dist, TensorVariable) or not isinstance(
            dist.owner.op, RandomVariable | SymbolicRandomVariable
        ):
            raise ValueError(
                f"Ordered dist must be a distribution created via the `.dist()` API, got {type(dist)}"
            )
        if dist.owner.op.ndim_supp > 0:
            raise NotImplementedError("Ordering of multivariate distributions not supported")
        if not all(
            all(param.type.broadcastable) for param in dist.owner.op.dist_params(dist.owner)
        ):
            raise ValueError("Ordered dist must be an IID variable")

        support_shape = get_support_shape_1d(
            support_shape=support_shape,
            shape=kwargs.get("shape", None),
        )
        if support_shape is not None:
            dist = change_dist_size(dist, support_shape)

        dist = pt.atleast_1d(dist)

        return super().dist([dist], **kwargs)


@_default_transform.register(OrderedRV)
def default_transform_ordered(op, rv):
    if rv.type.dtype.startswith("float"):
        return ordered
    else:
        return None


@_support_point.register(OrderedRV)
def support_point_ordered(op, rv, dist):
    # FIXME: This does not work with the default ordered transform
    # which maps [0, 0, 0] to [0, -inf, -inf].
    # return support_point(dist)
    return rv  # Draw from the prior
