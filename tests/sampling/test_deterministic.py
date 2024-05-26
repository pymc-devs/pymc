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
import numpy as np
import pytest

from numpy.testing import assert_allclose

from pymc.distributions import Normal
from pymc.model.core import Deterministic, Model
from pymc.sampling.deterministic import compute_deterministics
from pymc.sampling.forward import sample_prior_predictive

# Turn all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


def test_compute_deterministics():
    with Model(coords={"group": (0, 2, 4)}) as m:
        mu_raw = Normal("mu_raw", 0, 1, dims="group")
        mu = Deterministic("mu", mu_raw.cumsum(), dims="group")

        sigma_raw = Normal("sigma_raw", 0, 1)
        sigma = Deterministic("sigma", sigma_raw.exp())

    dataset = sample_prior_predictive(
        samples=5, model=m, var_names=["mu_raw", "sigma_raw"], random_seed=22
    ).prior

    # Test default
    with m:
        all_dets = compute_deterministics(dataset)
    assert set(all_dets.data_vars.variables) == {"mu", "sigma"}
    assert all_dets["mu"].dims == ("chain", "draw", "group")
    assert all_dets["sigma"].dims == ("chain", "draw")
    assert_allclose(all_dets["mu"], dataset["mu_raw"].cumsum("group"))
    assert_allclose(all_dets["sigma"], np.exp(dataset["sigma_raw"]))

    # Test custom arguments
    extended_with_mu = compute_deterministics(
        dataset,
        var_names=["mu"],
        merge_dataset=True,
        model=m,
        compile_kwargs={"mode": "FAST_COMPILE"},
        progressbar=False,
    )
    assert set(extended_with_mu.data_vars.variables) == {"mu_raw", "sigma_raw", "mu"}
    assert extended_with_mu["mu"].dims == ("chain", "draw", "group")
    assert_allclose(extended_with_mu["mu"], dataset["mu_raw"].cumsum("group"))

    only_sigma = compute_deterministics(dataset, var_names=["sigma"], model=m, progressbar=False)
    assert set(only_sigma.data_vars.variables) == {"sigma"}
    assert only_sigma["sigma"].dims == ("chain", "draw")
    assert_allclose(only_sigma["sigma"], np.exp(dataset["sigma_raw"]))


def test_docstring_example():
    import pymc as pm

    with pm.Model(coords={"group": (0, 2, 4)}) as m:
        mu_raw = pm.Normal("mu_raw", 0, 1, dims="group")
        mu = pm.Deterministic("mu", mu_raw.cumsum(), dims="group")

        trace = pm.sample(var_names=["mu_raw"], chains=2, tune=5, draws=5)

    assert "mu" not in trace.posterior

    with m:
        trace.posterior = pm.compute_deterministics(trace.posterior, merge_dataset=True)

    assert "mu" in trace.posterior
