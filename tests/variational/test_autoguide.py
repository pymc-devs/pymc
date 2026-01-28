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
import pytest

import pymc as pm

from pymc.variational.autoguide import AutoDiagonalNormal, AutoFullRankNormal, get_logp_logq

Parameter = pt.tensor


@pytest.fixture(scope="module")
def X_y_params():
    """Generate synthetic data for testing."""

    rng = np.random.default_rng(sum(map(ord, "autoguide_test")))

    alpha = rng.normal(loc=100, scale=10)
    beta = rng.normal(loc=0, scale=1, size=(10,))

    true_params = {
        "alpha": alpha,
        "beta": beta,
    }

    X_data = rng.normal(size=(100, 10))
    y_data = alpha + X_data @ beta

    return X_data, y_data, true_params


@pytest.fixture(scope="module")
def model(X_y_params):
    X_data, y_data, _ = X_y_params

    with pm.Model() as model:
        X = pm.Data("X", X_data)
        alpha = pm.Normal("alpha", 100, 10)
        beta = pm.Normal("beta", 0, 5, size=(10,))

        mu = alpha + X @ beta
        sigma = pm.Exponential("sigma", 1)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)

    return model


@pytest.fixture(scope="module")
def target_guide_model(X_y_params):
    X_data, *_ = X_y_params

    draws = pt.tensor("draws", shape=(), dtype="int64")

    with pm.Model() as guide_model:
        X = pm.Data("X", X_data)

        alpha_loc = Parameter("alpha_loc", shape=())
        alpha_scale = Parameter("alpha_scale", shape=())
        alpha_z = pm.Normal("alpha_z", mu=0, sigma=1, shape=(draws,))
        alpha = pm.Deterministic("alpha", alpha_loc + alpha_scale * alpha_z)

        beta_loc = Parameter("beta_loc", shape=(10,))
        beta_scale = Parameter("beta_scale", shape=(10,))
        beta_z = pm.Normal("beta_z", mu=0, sigma=1, shape=(draws, 10))
        beta = pm.Deterministic("beta", beta_loc + beta_scale * beta_z)

        sigma_loc = Parameter("sigma_loc", shape=())
        sigma_scale = Parameter("sigma_scale", shape=())
        sigma_z = pm.Normal(
            "sigma_z", 0, 1, shape=(draws,), transform=pm.distributions.transforms.log
        )
        sigma = pm.Deterministic("sigma", sigma_loc + sigma_scale * sigma_z)

    return guide_model


def test_diagonal_normal_autoguide(model, target_guide_model, X_y_params):
    guide_model = AutoDiagonalNormal(model)

    logp, logq = get_logp_logq(model, guide_model)
    logp_target, logq_target = get_logp_logq(model, target_guide_model)

    inputs = pm.inputvars(logp)
    target_inputs = pm.inputvars(logp_target)

    expected_locs = [f"{var}_loc" for var in ["alpha", "beta", "sigma"]]
    expected_scales = [f"{var}_scale" for var in ["alpha", "beta", "sigma"]]

    expected_inputs = expected_locs + expected_scales + ["draws"]
    name_to_input = {input.name: input for input in inputs}
    name_to_target_input = {input.name: input for input in target_inputs}

    assert all(input.name in expected_inputs for input in inputs), (
        "Guide inputs do not match expected inputs"
    )

    negative_elbo = (logq - logp).mean()
    negative_elbo_target = (logq_target - logp_target).mean()

    fn = pm.compile(
        [name_to_input[input] for input in expected_inputs], negative_elbo, random_seed=69420
    )
    fn_target = pm.compile(
        [name_to_target_input[input] for input in expected_inputs],
        negative_elbo_target,
        random_seed=69420,
    )

    test_inputs = {
        "alpha_loc": np.zeros(()),
        "alpha_scale": np.ones(()),
        "beta_loc": np.zeros(10),
        "beta_scale": np.ones(10),
        "sigma_loc": np.zeros(()),
        "sigma_scale": np.ones(()),
        "draws": 100,
    }

    np.testing.assert_allclose(fn(**test_inputs), fn_target(**test_inputs))


def test_full_mv_normal_guide(model, X_y_params):
    guide_model = AutoFullRankNormal(model)
    logp, logq = get_logp_logq(model, guide_model)
