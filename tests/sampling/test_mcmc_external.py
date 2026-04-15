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

import unittest.mock as mock

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from pymc import Data, Deterministic, HalfNormal, Model, Normal, sample
from pymc.progress_bar import NutpieProgressBarManager

pytestmark = pytest.mark.filterwarnings(
    "error",
    "ignore:There are not enough devices to run parallel chains:UserWarning",
    "ignore:os.fork\\(\\) was called:RuntimeWarning",
)


# temporarily skip nutpie
@pytest.mark.parametrize("nuts_sampler", ["pymc", "blackjax", "numpyro"])
# @pytest.mark.parametrize("nuts_sampler", ["pymc", "nutpie", "blackjax", "numpyro"])
def test_external_nuts_sampler(recwarn, nuts_sampler):
    if nuts_sampler != "pymc":
        pytest.importorskip(nuts_sampler)

    with Model():
        x = Normal("x", 100, 5)
        y = Data("y", [1, 2, 3, 4])
        Data("z", [100, 190, 310, 405])

        Normal("L", mu=x, sigma=0.1, observed=y)

        kwargs = {
            "nuts_sampler": nuts_sampler,
            "random_seed": 123,
            "chains": 2,
            "tune": 500,
            "draws": 500,
            "progressbar": False,
            "initvals": {"x": 0.0},
        }

        idata1 = sample(**kwargs)
        idata2 = sample(**kwargs)

        reference_kwargs = kwargs.copy()
        reference_kwargs["nuts_sampler"] = "pymc"
        idata_reference = sample(**reference_kwargs)

    assert "y" in idata1.constant_data
    assert "z" in idata1.constant_data
    assert "L" in idata1.observed_data
    assert idata1.posterior.chain.size == 2
    assert idata1.posterior.draw.size == 500
    np.testing.assert_array_equal(idata1.posterior.x, idata2.posterior.x)

    assert idata_reference.posterior.attrs.keys() == idata1.posterior.attrs.keys()


def test_step_args():
    pytest.importorskip("numpyro")

    with Model() as model:
        a = Normal("a")
        idata = sample(
            nuts_sampler="numpyro",
            target_accept=0.5,
            nuts={"max_tree_depth": 10},
            random_seed=1411,
            progressbar=False,
        )

    npt.assert_almost_equal(idata.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)


# temporarily skip nutpie
@pytest.mark.parametrize("nuts_sampler", ["pymc", "blackjax", "numpyro"])
# @pytest.mark.parametrize("nuts_sampler", ["pymc", "nutpie", "blackjax", "numpyro"])
def test_sample_var_names(nuts_sampler):
    if nuts_sampler != "pymc":
        pytest.importorskip(nuts_sampler)

    seed = 1234
    kwargs = {
        "chains": 1,
        "tune": 100,
        "draws": 100,
        "random_seed": seed,
        "progressbar": False,
        "compute_convergence_checks": False,
    }

    # Generate data
    rng = np.random.default_rng(seed)

    group = rng.choice(list("ABCD"), size=100)
    x = rng.normal(size=100)
    y = rng.normal(size=100)

    group_values, group_idx = np.unique(group, return_inverse=True)

    coords = {"group": group_values}

    # Create model
    with Model(coords=coords) as model:
        b_group = Normal("b_group", dims="group")
        b_x = Normal("b_x")
        mu = Deterministic("mu", b_group[group_idx] + b_x * x)
        sigma = HalfNormal("sigma")
        Normal("y", mu=mu, sigma=sigma, observed=y)

    free_RVs = [var.name for var in model.free_RVs]

    with model:
        # Sample with and without var_names, but always with the same seed
        idata_1 = sample(nuts_sampler=nuts_sampler, **kwargs)
        # Remove the last free RV from the sampling
        idata_2 = sample(nuts_sampler=nuts_sampler, var_names=free_RVs[:-1], **kwargs)

    assert "mu" in idata_1.posterior
    assert "mu" not in idata_2.posterior

    assert free_RVs[-1] in idata_1.posterior
    assert free_RVs[-1] not in idata_2.posterior

    for var in free_RVs[:-1]:
        assert var in idata_1.posterior
        assert var in idata_2.posterior

        xr.testing.assert_allclose(idata_1.posterior[var], idata_2.posterior[var])


def test_nutpie_progress_bar_manager_update():
    pb = NutpieProgressBarManager(chains=2, draws=10, tune=10, progressbar=False)
    pb._backend = mock.Mock()
    pb._show_progress = True  # force the update path even without a real backend

    cp0 = SimpleNamespace(
        finished_draws=5,
        total_draws=20,
        divergences=0,
        step_size=0.5,
        latest_num_steps=3,
    )
    cp1 = SimpleNamespace(
        finished_draws=4,
        total_draws=20,
        divergences=1,
        step_size=0.4,
        latest_num_steps=7,
    )
    pb.update([cp0, cp1])
    assert pb._backend.update.call_count == 2
    first_call = pb._backend.update.call_args_list[0].kwargs
    assert first_call["task_id"] == 0
    assert first_call["advance"] == 5
    assert first_call["stats"]["divergences"] == 0
    second_call = pb._backend.update.call_args_list[1].kwargs
    assert second_call["task_id"] == 1
    assert second_call["advance"] == 4
    assert second_call["failing"] is True

    # A second update only advances by the delta since the previous call.
    cp0.finished_draws = 20
    cp1.finished_draws = 20
    pb._backend.update.reset_mock()
    pb.update([cp0, cp1])
    deltas = [c.kwargs["advance"] for c in pb._backend.update.call_args_list]
    is_last_flags = [c.kwargs["is_last"] for c in pb._backend.update.call_args_list]
    assert deltas == [15, 16]
    assert is_last_flags == [True, True]


def test_nutpie_end_to_end():
    # Released nutpie 0.16.8 references `arviz.InferenceData` which arviz 1.0 removed,
    # so `import nutpie` raises AttributeError on the current CI matrix. Skip until a
    # nutpie release compatible with arviz 1.0 ships.
    try:
        import nutpie  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("nutpie unavailable or incompatible with the installed arviz")
    with Model() as m:
        HalfNormal("sigma")
        Normal("mu")
        Normal("y", mu=0, sigma=1, observed=[1.0, 2.0, 3.0])
        idata = sample(
            nuts_sampler="nutpie",
            tune=20,
            draws=20,
            chains=2,
            progressbar=False,
            random_seed=1411,
        )
    assert {"posterior", "sample_stats", "observed_data"} <= set(idata.children)
    assert set(idata.posterior.data_vars) == {"mu", "sigma"}
    assert idata.posterior.sizes == {"chain": 2, "draw": 20}
