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

from pymc.dims.distributions.core import DimDistribution, copy_docstring, expand_dist_dims
from pymc.distributions.censored import Censored as RegularCensored


@copy_docstring(RegularCensored)
class Censored(DimDistribution):
    @classmethod
    def dist(cls, dist, *, lower=None, upper=None, dim_lengths, **kwargs):
        if lower is None:
            lower = -np.inf
        if upper is None:
            upper = np.inf
        return super().dist([dist, lower, upper], dim_lengths=dim_lengths, **kwargs)

    @classmethod
    def xrv_op(cls, dist, lower, upper, core_dims=None, extra_dims=None, rng=None):
        if extra_dims is None:
            extra_dims = {}

        dist = cls._as_xtensor(dist)
        lower = cls._as_xtensor(lower)
        upper = cls._as_xtensor(upper)

        # Any dimensions in extra_dims, or only present in lower, upper,
        # must propagate back to the dist as `extra_dims`
        bounds_sizes = lower.sizes | upper.sizes
        dist_dims_set = set(dist.dims)
        extra_dist_dims = extra_dims | {
            dim: size for dim, size in bounds_sizes.items() if dim not in dist_dims_set
        }
        if extra_dist_dims:
            dist = expand_dist_dims(dist, extra_dist_dims)

        # Probability is inferred from the clip operation
        # TODO: Make this a SymbolicRandomVariable that can itself be resized
        return dist.clip(lower, upper)
