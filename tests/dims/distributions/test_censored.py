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
import pytest

from pytensor.xtensor import as_xtensor
from pytensor.xtensor.shape import Transpose
from pytensor.xtensor.vectorization import XRV

import pymc.distributions as regular_distributions

from pymc.dims import Censored, Normal
from pymc.model.core import Model
from tests.dims.utils import assert_equivalent_logp_graph, assert_equivalent_random_graph


@pytest.mark.parametrize("lower", [None, -1])
@pytest.mark.parametrize("upper", [None, 1])
def test_censored_basic(lower, upper):
    coords = {"space": range(3), "time": range(4)}

    with Model(coords=coords) as model:
        dist = Normal.dist(np.pi, np.e)
        Censored("y", dist, lower=lower, upper=upper, dims=("space", "time"))

    with Model(coords=coords) as reference_model:
        dist = regular_distributions.Normal.dist(np.pi, np.e)
        regular_distributions.Censored(
            "y", dist=dist, lower=lower, upper=upper, dims=("space", "time")
        )

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_censored_dims():
    """Test that both censored (and the underlying dist) have all the implied and explicit dims."""
    coords = {
        "a": range(3),
        "b": range(4),
        "c": range(5),
        "d": range(6),
    }
    with Model(coords=coords) as model:
        dist = Normal.dist(
            mu=as_xtensor([0, 1, 2], dims=("a",)),
            sigma=as_xtensor([1, 2, 3], dims=("b",)),
            dim_lengths={"c": model.dim_lengths["c"]},
        )
        assert set(dist.dims) == {"c", "a", "b"}

        c0 = Censored("c0", dist)
        assert c0.dims == ("c", "a", "b")
        c0_dist = c0.owner.inputs[0]
        assert isinstance(c0_dist.owner.op, XRV)
        assert c0_dist.dims == ("c", "a", "b")

        c1 = Censored("c1", dist, dims=("a", "b", "c"))
        assert c1.dims == ("a", "b", "c")
        assert isinstance(c1.owner.op, Transpose)
        c1_dist = c1.owner.inputs[0].owner.inputs[0]
        assert isinstance(c1_dist.owner.op, XRV)
        assert c1_dist.dims == ("c", "a", "b")

        c2 = Censored("c2", dist, dims=(..., "d"))
        assert c2.dims == ("c", "a", "b", "d")
        assert isinstance(c1.owner.op, Transpose)
        c2_dist = c2.owner.inputs[0].owner.inputs[0]
        assert isinstance(c2_dist.owner.op, XRV)
        assert c2_dist.dims == ("d", "c", "a", "b")

        lower = as_xtensor(np.zeros((6, 5)), dims=("d", "c"))
        c3 = Censored("c3", dist, lower=lower)
        assert c3.dims == ("d", "c", "a", "b")
        c3_dist = c3.owner.inputs[0]
        assert isinstance(c3_dist.owner.op, XRV)
        assert c3_dist.dims == ("d", "c", "a", "b")
