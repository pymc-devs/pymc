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
import pytensor.tensor as pt
import pytensor.xtensor.math as ptxm
import pytest

from pytensor.xtensor import as_xtensor
from pytensor.xtensor.shape import Transpose
from pytensor.xtensor.vectorization import XRV

import pymc.distributions as regular_distributions

from pymc import draw as pm_draw
from pymc.dims import Censored, CustomDist, Normal
from pymc.model.core import Model
from tests.dims.utils import assert_equivalent_logp_graph, assert_equivalent_random_graph

pytestmark = pytest.mark.filterwarnings("error")


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

    # Dead branches of the support point switch constant-fold -inf + inf
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(
            model.initial_point()["y"],
            reference_model.initial_point()["y"],
        )


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


def test_censored_custom_dist():
    """Factory distributions compose with an overridden CustomDist through its Op."""
    from scipy import stats

    def normal_dist(mu, sigma, extra_dims):
        return Normal.dist(mu, sigma, dim_lengths=extra_dims)

    def normal_logp(value, mu, sigma):
        value, mu, sigma = value.values, mu.values, sigma.values
        return -0.5 * ((value - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * np.pi))

    def normal_logcdf(value, mu, sigma):
        value, mu, sigma = value.values, mu.values, sigma.values
        return pt.log(pt.erf((value - mu) / (sigma * pt.sqrt(2.0))) + 1.0) - pt.log(2.0)

    coords = {"city": range(5)}
    with Model(coords=coords) as model:
        base = CustomDist.dist(0.0, 1.0, dist=normal_dist, logp=normal_logp, logcdf=normal_logcdf)
        # The extra dim is added by rebuilding the CustomDist through its Op
        Censored("y", base, lower=-1.0, upper=1.0, dims="city")

    draws = pm_draw(model["y"], draws=100, random_seed=1)
    assert draws.shape == (100, 5)
    assert ((draws >= -1) & (draws <= 1)).all()

    test_value = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    logp_value = model.compile_logp()({"y": test_value})
    ref = stats.norm(0, 1)
    expected = np.log(ref.cdf(-1)) + ref.logpdf([-0.5, 0.0, 0.5]).sum() + np.log(ref.sf(1))
    np.testing.assert_allclose(logp_value, expected)


def test_censored_custom_dist_derived():
    """Censored resizes a compound CustomDist without overrides through its Op."""

    def logitnormal_dist(mu, sigma, extra_dims):
        return ptxm.sigmoid(Normal.dist(mu, sigma, dim_lengths=extra_dims))

    coords = {"city": range(5)}
    with Model(coords=coords) as model:
        base = CustomDist.dist(0.0, 1.0, dist=logitnormal_dist)
        Censored("y", base, lower=0.2, upper=0.8, dims="city")

    draws = pm_draw(model["y"], draws=100, random_seed=1)
    assert draws.shape == (100, 5)
    assert ((draws >= 0.2) & (draws <= 0.8)).all()

    # The support point comes from the bounds, like in regular Censored
    np.testing.assert_allclose(model.initial_point()["y"], np.full(5, 0.5))
