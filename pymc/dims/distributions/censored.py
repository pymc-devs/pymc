#   Copyright 2026 - present The PyMC Developers
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

from pytensor.xtensor import as_xtensor, broadcast
from pytensor.xtensor.random import shared_rng
from pytensor.xtensor.random.type import xrandom_generator_type

from pymc.dims.distributions.core import (
    DimDistribution,
    DimSymbolicRandomVariable,
    copy_docstring,
    expand_dist_dims,
)
from pymc.distributions.censored import Censored as RegularCensored
from pymc.distributions.censored import support_point_censored
from pymc.distributions.distribution import _support_point


class DimCensoredRV(DimSymbolicRandomVariable):
    """Censored distribution on XTensorVariables.

    The logp is derived from the inner clip graph. The randomness belongs to
    the censored dist input, so the RNG input is passed through unchanged.
    """

    inline_logprob = True
    _print_name = ("Censored", "\\operatorname{Censored}")

    @classmethod
    def rv_op(
        cls,
        dist,
        lower,
        upper,
        *,
        core_dims=None,
        extra_dims=None,
        rng=None,
        # The next rng is always returned; the argument exists only while
        # the pytensor XRV constructors transition to doing the same
        return_next_rng: bool = True,
    ):
        assert return_next_rng, "return_next_rng=False is not supported"
        dist = DimDistribution._as_xtensor(dist)
        lower = DimDistribution._as_xtensor(lower)
        upper = DimDistribution._as_xtensor(upper)

        # Any dimensions in extra_dims, or only present in lower, upper,
        # must propagate back to the dist as `extra_dims`
        bounds_sizes = lower.sizes | upper.sizes
        dist_dims_set = set(dist.dims)
        extra_dist_dims = (extra_dims or {}) | {
            dim: size for dim, size in bounds_sizes.items() if dim not in dist_dims_set
        }
        if extra_dist_dims:
            dist = expand_dist_dims(dist, extra_dist_dims)

        # Censoring is achieved by clipping the dist between lower and upper.
        # The randomness belongs to the dist input, so the RNG is passed through
        dummy_rng = xrandom_generator_type("rng")
        op = cls(
            inputs=[dist, lower, upper, dummy_rng],
            outputs=[dist.clip(lower, upper), dummy_rng],
        )
        if rng is None:
            rng = shared_rng(seed=None)
        censored, next_rng = op(dist, lower, upper, rng)
        return next_rng, censored


@_support_point.register(DimCensoredRV)
def dim_censored_support_point(op, rv, dist, lower, upper, rng):
    # Align the inputs by name and reuse the regular CensoredRV implementation
    dist, lower, upper = broadcast(dist, lower, upper)
    sp = support_point_censored(op, rv.values, dist.values, lower.values, upper.values)
    return as_xtensor(sp, dims=dist.type.dims).transpose(*rv.type.dims)


@copy_docstring(RegularCensored)
class Censored(DimDistribution):
    xrv_op = DimCensoredRV.rv_op

    @classmethod
    def dist(cls, dist, *, lower=None, upper=None, dim_lengths, **kwargs):
        if lower is None:
            lower = -np.inf
        if upper is None:
            upper = np.inf
        return super().dist([dist, lower, upper], dim_lengths=dim_lengths, **kwargs)
