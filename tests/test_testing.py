#   Copyright 2024 - present The PyMC Developers
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
from contextlib import ExitStack as does_not_raise

import pytest

import pymc as pm

from pymc.testing import Domain, mock_sample, mock_sample_setup_and_teardown
from tests.models import simple_normal


@pytest.mark.parametrize(
    "values, edges, expectation",
    [
        ([], None, pytest.raises(IndexError)),
        ([], (0, 0), pytest.raises(ValueError)),
        ([0], None, pytest.raises(ValueError)),
        ([0], (0, 0), does_not_raise()),
        ([-1, 1], None, pytest.raises(ValueError)),
        ([-1, 0, 1], None, does_not_raise()),
    ],
)
def test_domain(values, edges, expectation):
    with expectation:
        Domain(values, edges=edges)


@pytest.mark.parametrize(
    "args, kwargs, expected_size",
    [
        pytest.param((), {}, (1, 10), id="default"),
        pytest.param((100,), {}, (1, 100), id="positional-draws"),
        pytest.param((), {"draws": 100}, (1, 100), id="keyword-draws"),
        pytest.param((100,), {"chains": 6}, (6, 100), id="chains"),
    ],
)
def test_mock_sample(args, kwargs, expected_size) -> None:
    expected_chains, expected_draws = expected_size
    _, model, _ = simple_normal(bounded_prior=True)

    with model:
        idata = mock_sample(*args, **kwargs)

    assert "posterior" in idata
    assert "observed_data" in idata
    assert "prior" not in idata
    assert "posterior_predictive" not in idata
    assert "sample_stats" not in idata

    assert idata.posterior.sizes == {"chain": expected_chains, "draw": expected_draws}


mock_pymc_sample = pytest.fixture(scope="function")(mock_sample_setup_and_teardown)


@pytest.fixture(scope="function")
def dummy_model() -> pm.Model:
    with pm.Model() as model:
        pm.Flat("flat")
        pm.HalfFlat("half_flat")

    return model


def test_fixture(mock_pymc_sample, dummy_model) -> None:
    with dummy_model:
        idata = pm.sample()

    posterior = idata.posterior
    assert posterior.sizes == {"chain": 1, "draw": 10}
    assert (posterior["half_flat"] >= 0).all()
