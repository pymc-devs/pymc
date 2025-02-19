#   Copyright 2025 - present The PyMC Developers
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
from contextlib import nullcontext

import numpy as np
import pytest

import pymc as pm

from pymc import DirichletMultinomial, MvStudentT
from pymc.model.transform.optimization import freeze_dims_and_data


@pytest.fixture(params=["FAST_RUN", "JAX", "NUMBA"])
def mode(request):
    mode_param = request.param
    if mode_param != "FAST_RUN":
        pytest.importorskip(mode_param.lower())
    return mode_param


def test_dirichlet_multinomial(mode):
    """Test we can draw from a DM in the JAX backend if the shape is constant."""
    dm = DirichletMultinomial.dist(n=5, a=np.eye(3) * 1e6 + 0.01)
    dm_draws = pm.draw(dm, mode=mode)
    np.testing.assert_equal(dm_draws, np.eye(3) * 5)


def test_dirichlet_multinomial_dims(mode):
    """Test we can draw from a DM with a shape defined by dims in the JAX backend,
    after freezing those dims.
    """
    with pm.Model(coords={"trial": range(3), "item": range(3)}) as m:
        dm = DirichletMultinomial("dm", n=5, a=np.eye(3) * 1e6 + 0.01, dims=("trial", "item"))

    # JAX does not allow us to JIT a function with dynamic shape
    expected_ctxt = pytest.raises(TypeError) if mode == "JAX" else nullcontext()
    with expected_ctxt:
        pm.draw(dm, mode=mode)

    # Should be fine after freezing the dims that specify the shape
    frozen_dm = freeze_dims_and_data(m)["dm"]
    dm_draws = pm.draw(frozen_dm, mode=mode)
    np.testing.assert_equal(dm_draws, np.eye(3) * 5)


def test_mvstudentt(mode):
    mvt = MvStudentT.dist(nu=100, mu=[1, 2, 3], scale=np.eye(3) * [0.01, 1, 100], shape=(10_000, 3))
    draws = pm.draw(mvt, mode=mode)
    np.testing.assert_allclose(draws.mean(0), [1, 2, 3], rtol=0.1)
    np.testing.assert_allclose(draws.std(0), np.sqrt([0.01, 1, 100]), rtol=0.1)


def test_repeated_arguments(mode):
    # Regression test for a failure in Numba mode when a RV had repeated arguments
    v = 0.5 * 1e5
    x = pm.Beta.dist(v, v)
    x_draw = pm.draw(x, mode=mode)
    np.testing.assert_allclose(x_draw, 0.5, rtol=0.01)
