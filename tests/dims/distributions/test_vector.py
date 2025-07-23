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
import numpy as np
import pytensor.tensor as pt

from pytensor.xtensor import as_xtensor

import pymc.distributions as regular_distributions

from pymc import Model
from pymc.dims import Categorical, MvNormal, ZeroSumNormal
from tests.dims.utils import assert_equivalent_logp_graph, assert_equivalent_random_graph


def test_categorical():
    coords = {"a": range(3), "b": range(4)}
    p = pt.as_tensor([0.1, 0.2, 0.3, 0.4])
    p_xr = as_xtensor(p, dims=("b",))

    with Model(coords=coords) as model:
        Categorical("x", p=p_xr, core_dims="b", dims=("a",))
        Categorical("y", logit_p=p_xr, core_dims="b", dims=("a",))

    with Model(coords=coords) as reference_model:
        regular_distributions.Categorical("x", p=p, dims=("a",))
        regular_distributions.Categorical("y", logit_p=p, dims=("a",))

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_mvnormal():
    coords = {"a": range(3), "b": range(2)}
    mu = pt.as_tensor([1, 2])
    cov = pt.as_tensor([[1, 0.5], [0.5, 2]])
    chol = pt.as_tensor([[1, 0], [0.5, np.sqrt(1.75)]])

    mu_xr = as_xtensor(mu, dims=("b",))
    cov_xr = as_xtensor(cov, dims=("b", "b'"))
    chol_xr = as_xtensor(chol, dims=("b", "b'"))

    with Model(coords=coords) as model:
        MvNormal("x", mu=mu_xr, cov=cov_xr, core_dims=("b", "b'"), dims=("a", "b"))
        MvNormal("y", mu=mu_xr, chol=chol_xr, core_dims=("b", "b'"), dims=("a", "b"))

    with Model(coords=coords) as reference_model:
        regular_distributions.MvNormal("x", mu=mu, cov=cov, dims=("a", "b"))
        regular_distributions.MvNormal("y", mu=mu, chol=chol, dims=("a", "b"))

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_zerosumnormal():
    coords = {"a": range(3), "b": range(2)}
    with Model(coords=coords) as model:
        ZeroSumNormal("x", core_dims=("b",), dims=("a", "b"))
        ZeroSumNormal("y", sigma=3, core_dims=("b",), dims=("a", "b"))
        ZeroSumNormal("z", core_dims=("a", "b"), dims=("a", "b"))

    with Model(coords=coords) as reference_model:
        regular_distributions.ZeroSumNormal("x", dims=("a", "b"))
        regular_distributions.ZeroSumNormal("y", sigma=3, n_zerosum_axes=1, dims=("a", "b"))
        regular_distributions.ZeroSumNormal("z", n_zerosum_axes=2, dims=("a", "b"))

    assert_equivalent_random_graph(model, reference_model)
    # Logp is correct, but we have join(..., -1) and join(..., 1), that don't get canonicalized to the same
    # Should work once https://github.com/pymc-devs/pytensor/issues/1505 is fixed
    # assert_equivalent_logp_graph(model, reference_model)
