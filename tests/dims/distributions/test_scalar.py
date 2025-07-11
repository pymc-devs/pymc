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
from pymc import Model
from pymc import distributions as regular_distributions
from pymc.dims import (
    Beta,
    Cauchy,
    Exponential,
    Flat,
    Gamma,
    HalfCauchy,
    HalfFlat,
    HalfNormal,
    HalfStudentT,
    InverseGamma,
    Laplace,
    LogNormal,
    Normal,
    StudentT,
)
from tests.dims.utils import assert_equivalent_logp_graph, assert_equivalent_random_graph


def test_flat():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        Flat("x", dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.Flat("x", dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_halfflat():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        HalfFlat("x", dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.HalfFlat("x", dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_normal():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        Normal("x", dims="a")
        Normal("y", mu=2, sigma=3, dims="a")
        Normal("z", mu=-2, tau=3, dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.Normal("x", dims="a")
        regular_distributions.Normal("y", mu=2, sigma=3, dims="a")
        regular_distributions.Normal("z", mu=-2, tau=3, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_halfnormal():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        HalfNormal("x", dims="a")
        HalfNormal("y", sigma=3, dims="a")
        HalfNormal("z", tau=3, dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.HalfNormal("x", dims="a")
        regular_distributions.HalfNormal("y", sigma=3, dims="a")
        regular_distributions.HalfNormal("z", tau=3, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_lognormal():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        LogNormal("x", dims="a")
        LogNormal("y", mu=2, sigma=3, dims="a")
        LogNormal("z", mu=-2, tau=3, dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.LogNormal("x", dims="a")
        regular_distributions.LogNormal("y", mu=2, sigma=3, dims="a")
        regular_distributions.LogNormal("z", mu=-2, tau=3, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_studentt():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        StudentT("x", nu=1, dims="a")
        StudentT("y", nu=1, mu=2, sigma=3, dims="a")
        StudentT("z", nu=1, mu=-2, lam=3, dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.StudentT("x", nu=1, dims="a")
        regular_distributions.StudentT("y", nu=1, mu=2, sigma=3, dims="a")
        regular_distributions.StudentT("z", nu=1, mu=-2, lam=3, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_halfstudentt():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        HalfStudentT("x", nu=1, dims="a")
        HalfStudentT("y", nu=1, sigma=3, dims="a")
        HalfStudentT("z", nu=1, lam=3, dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.HalfStudentT("x", nu=1, dims="a")
        regular_distributions.HalfStudentT("y", nu=1, sigma=3, dims="a")
        regular_distributions.HalfStudentT("z", nu=1, lam=3, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_cauchy():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        Cauchy("x", alpha=1, beta=2, dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.Cauchy("x", alpha=1, beta=2, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_halfcauchy():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        HalfCauchy("x", beta=2, dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.HalfCauchy("x", beta=2, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_beta():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        Beta("w", alpha=1, beta=1, dims="a")
        Beta("x", mu=0.5, sigma=0.1, dims="a")
        Beta("y", mu=0.5, nu=10, dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.Beta("w", alpha=1, beta=1, dims="a")
        regular_distributions.Beta("x", mu=0.5, sigma=0.1, dims="a")
        regular_distributions.Beta("y", mu=0.5, nu=10, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_laplace():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        Laplace("x", dims="a")
        Laplace("y", mu=1, b=2, dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.Laplace("x", mu=0, b=1, dims="a")
        regular_distributions.Laplace("y", mu=1, b=2, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_exponential():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        Exponential("x", dims="a")
        Exponential("y", lam=2, dims="a")
        Exponential("z", scale=3, dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.Exponential("x", dims="a")
        regular_distributions.Exponential("y", lam=2, dims="a")
        regular_distributions.Exponential("z", scale=3, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_gamma():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        # Gamma("w", alpha=1, beta=1, dims="a")
        Gamma("x", mu=2, sigma=3, dims="a")

    with Model(coords=coords) as reference_model:
        # regular_distributions.Gamma("w", alpha=1, beta=1, dims="a")
        regular_distributions.Gamma("x", mu=2, sigma=3, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)


def test_inverse_gamma():
    coords = {"a": range(3)}
    with Model(coords=coords) as model:
        InverseGamma("w", alpha=1, beta=1, dims="a")
        InverseGamma("x", mu=2, sigma=3, dims="a")

    with Model(coords=coords) as reference_model:
        regular_distributions.InverseGamma("w", alpha=1, beta=1, dims="a")
        regular_distributions.InverseGamma("x", mu=2, sigma=3, dims="a")

    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)
