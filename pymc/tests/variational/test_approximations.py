#   Copyright 2023 The PyMC Developers
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
import pytensor
import pytest

import pymc as pm
import pymc.tests.models as models

from pymc.variational.approximations import Empirical, MeanField


def test_empirical_does_not_support_inference_data():
    with models.another_simple_model():
        step = pm.Metropolis()
        trace = pm.sample(100, step=step, chains=1, tune=0, return_inferencedata=True)
        with pytest.raises(NotImplementedError, match="return_inferencedata=False"):
            Empirical(trace)


def test_empirical_from_trace():
    with models.another_simple_model():
        step = pm.Metropolis()
        trace = pm.sample(100, step=step, chains=1, tune=0, return_inferencedata=False)
        emp = Empirical(trace)
        assert emp.histogram.shape[0].eval() == 100
        trace = pm.sample(100, step=step, chains=4, tune=0, return_inferencedata=False)
        emp = Empirical(trace)
        assert emp.histogram.shape[0].eval() == 400


def test_elbo():
    mu0 = 1.5
    sigma = 1.0
    y_obs = np.array([1.6, 1.4])

    post_mu = np.array([1.88], dtype=pytensor.config.floatX)
    post_sigma = np.array([1], dtype=pytensor.config.floatX)
    # Create a model for test
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=mu0, sigma=sigma)
        pm.Normal("y", mu=mu, sigma=1, observed=y_obs)

    # Create variational gradient tensor
    mean_field = MeanField(model=model)
    with pytensor.config.change_flags(compute_test_value="off"):
        elbo = -pm.operators.KL(mean_field)()(10000)

    mean_field.shared_params["mu"].set_value(post_mu)
    mean_field.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

    f = pytensor.function([], elbo)
    elbo_mc = f()

    # Exact value
    elbo_true = -0.5 * (
        3
        + 3 * post_mu**2
        - 2 * (y_obs[0] + y_obs[1] + mu0) * post_mu
        + y_obs[0] ** 2
        + y_obs[1] ** 2
        + mu0**2
        + 3 * np.log(2 * np.pi)
    ) + 0.5 * (np.log(2 * np.pi) + 1)
    np.testing.assert_allclose(elbo_mc, elbo_true, rtol=0, atol=1e-1)


@pytest.mark.parametrize("aux_total_size", range(2, 10, 3))
def test_scale_cost_to_minibatch_works(aux_total_size):
    mu0 = 1.5
    sigma = 1.0
    y_obs = np.array([1.6, 1.4])
    beta = len(y_obs) / float(aux_total_size)

    # TODO: pytensor_config
    # with pm.Model(pytensor_config=dict(floatX='float64')):
    # did not not work as expected
    # there were some numeric problems, so float64 is forced
    with pytensor.config.change_flags(floatX="float64", warn_float64="ignore"):
        assert pytensor.config.floatX == "float64"
        assert pytensor.config.warn_float64 == "ignore"

        post_mu = np.array([1.88], dtype=pytensor.config.floatX)
        post_sigma = np.array([1], dtype=pytensor.config.floatX)

        with pm.Model():
            mu = pm.Normal("mu", mu=mu0, sigma=sigma)
            pm.Normal("y", mu=mu, sigma=1, observed=y_obs, total_size=aux_total_size)
            # Create variational gradient tensor
            mean_field_1 = MeanField()
            assert mean_field_1.scale_cost_to_minibatch
            mean_field_1.shared_params["mu"].set_value(post_mu)
            mean_field_1.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

            with pytensor.config.change_flags(compute_test_value="off"):
                elbo_via_total_size_scaled = -pm.operators.KL(mean_field_1)()(10000)

        with pm.Model():
            mu = pm.Normal("mu", mu=mu0, sigma=sigma)
            pm.Normal("y", mu=mu, sigma=1, observed=y_obs, total_size=aux_total_size)
            # Create variational gradient tensor
            mean_field_2 = MeanField()
            assert mean_field_1.scale_cost_to_minibatch
            mean_field_2.scale_cost_to_minibatch = False
            assert not mean_field_2.scale_cost_to_minibatch
            mean_field_2.shared_params["mu"].set_value(post_mu)
            mean_field_2.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

        with pytensor.config.change_flags(compute_test_value="off"):
            elbo_via_total_size_unscaled = -pm.operators.KL(mean_field_2)()(10000)

        np.testing.assert_allclose(
            elbo_via_total_size_unscaled.eval(),
            elbo_via_total_size_scaled.eval() * pm.floatX(1 / beta),
            rtol=0.02,
            atol=1e-1,
        )


@pytest.mark.parametrize("aux_total_size", range(2, 10, 3))
def test_elbo_beta_kl(aux_total_size):
    mu0 = 1.5
    sigma = 1.0
    y_obs = np.array([1.6, 1.4])
    beta = len(y_obs) / float(aux_total_size)

    with pytensor.config.change_flags(floatX="float64", warn_float64="ignore"):
        post_mu = np.array([1.88], dtype=pytensor.config.floatX)
        post_sigma = np.array([1], dtype=pytensor.config.floatX)

        with pm.Model():
            mu = pm.Normal("mu", mu=mu0, sigma=sigma)
            pm.Normal("y", mu=mu, sigma=1, observed=y_obs, total_size=aux_total_size)
            # Create variational gradient tensor
            mean_field_1 = MeanField()
            mean_field_1.scale_cost_to_minibatch = True
            mean_field_1.shared_params["mu"].set_value(post_mu)
            mean_field_1.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

            with pytensor.config.change_flags(compute_test_value="off"):
                elbo_via_total_size_scaled = -pm.operators.KL(mean_field_1)()(10000)

        with pm.Model():
            mu = pm.Normal("mu", mu=mu0, sigma=sigma)
            pm.Normal("y", mu=mu, sigma=1, observed=y_obs)
            # Create variational gradient tensor
            mean_field_3 = MeanField()
            mean_field_3.shared_params["mu"].set_value(post_mu)
            mean_field_3.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

            with pytensor.config.change_flags(compute_test_value="off"):
                elbo_via_beta_kl = -pm.operators.KL(mean_field_3, beta=beta)()(10000)

        np.testing.assert_allclose(
            elbo_via_total_size_scaled.eval(), elbo_via_beta_kl.eval(), rtol=0, atol=1e-1
        )


def test_seeding_advi_fit():
    with pm.Model():
        x = pm.Normal("x", 0, 10, initval="prior")
        approx1 = pm.fit(
            random_seed=42, n=10, method="advi", obj_optimizer=pm.adagrad_window, progressbar=False
        )
        approx2 = pm.fit(
            random_seed=42, n=10, method="advi", obj_optimizer=pm.adagrad_window, progressbar=False
        )
        np.testing.assert_allclose(approx1.mean.eval(), approx2.mean.eval())
