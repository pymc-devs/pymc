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

import pymc as pm

pytest.importorskip("jax")

# Importing pymc.sampling.jax sets XLA_FLAGS (host device count for pmap-based
# parallel chains). It must be imported before blackjax initializes the jax
# backend — including for the other test modules collected in this session.
import pymc.sampling.jax

blackjax = pytest.importorskip("blackjax")

from pymc.sampling.external.blackjax import Blackjax  # noqa: E402


@pytest.fixture(scope="module")
def gaussian_model_data():
    rng = np.random.default_rng(19)
    return rng.normal(0.5, 1.2, size=200)


def make_model(data):
    with pm.Model(coords={"idx": range(3)}) as model:
        mu = pm.Normal("mu")
        sigma = pm.HalfNormal("sigma")
        noise = pm.Normal("noise", dims="idx")
        pm.Normal("y", mu + noise.mean(), sigma, observed=data)
    return model


@pytest.mark.parametrize(
    "algorithm,kwargs",
    [
        ("nuts", {"target_accept": 0.9}),
        ("hmc", {"num_integration_steps": 16}),
        ("mclmc", {}),
        ("mala", {"step_size": 0.02}),
        ("barker", {"step_size": 0.2}),
    ],
)
def test_blackjax_algorithms(gaussian_model_data, algorithm, kwargs):
    model = make_model(gaussian_model_data)
    with model:
        idata = pm.sample(
            external_sampler=Blackjax(algorithm, chain_method="vectorized", **kwargs),
            chains=2,
            tune=700,
            draws=500,
            progressbar=False,
            random_seed=42,
            compute_convergence_checks=False,
        )

    assert idata.posterior.sizes == {"chain": 2, "draw": 500, "idx": 3}
    assert "lp" in idata.sample_stats
    # Gradient-based tuned algorithms should recover the true parameters
    if algorithm in ("nuts", "hmc", "mclmc"):
        np.testing.assert_allclose(idata.posterior["mu"].mean(), 0.5, atol=0.2)
        np.testing.assert_allclose(idata.posterior["sigma"].mean(), 1.2, atol=0.2)
        if algorithm != "mclmc":
            assert "diverging" in idata.sample_stats


def test_blackjax_algorithm_object(gaussian_model_data):
    model = make_model(gaussian_model_data)
    sampler = Blackjax(blackjax.nuts, model=model, chain_method="vectorized")
    assert sampler.algorithm_name == "nuts"
    assert sampler.adaptation == "window"


def test_blackjax_stat_renames(gaussian_model_data):
    model = make_model(gaussian_model_data)
    with model:
        idata = pm.sample(
            external_sampler=Blackjax(chain_method="vectorized"),
            chains=1,
            tune=300,
            draws=100,
            progressbar=False,
            random_seed=42,
            compute_convergence_checks=False,
        )
    stats = set(idata.sample_stats.data_vars)
    assert {"diverging", "energy", "tree_depth", "n_steps", "acceptance_rate", "lp"} <= stats


def test_blackjax_get_kwargs():
    with pm.Model():
        pm.Normal("x", shape=3)
        kwargs = Blackjax(target_accept=0.95).get_kwargs()
    assert kwargs["algorithm"]["step_size"] == "<required>"
    assert kwargs["algorithm"]["max_num_doublings"] == 10
    assert kwargs["adaptation"]["target_acceptance_rate"] == 0.95


def test_blackjax_kwargs_routing():
    with pm.Model():
        pm.Normal("x", shape=3)
        sampler = Blackjax(
            "nuts", max_num_doublings=8, is_mass_matrix_diagonal=False, target_accept=0.9
        )
    assert sampler.algorithm_kwargs == {"max_num_doublings": 8}
    assert sampler.adaptation_kwargs == {
        "is_mass_matrix_diagonal": False,
        "target_acceptance_rate": 0.9,
    }


class TestBlackjaxErrors:
    def setup_method(self):
        self.model = pm.Model()
        with self.model:
            pm.Normal("x", shape=3)

    def test_unknown_algorithm(self):
        with self.model, pytest.raises(ValueError, match="Unknown blackjax algorithm 'nutz'"):
            Blackjax("nutz")

    def test_vi_algorithm(self):
        with self.model, pytest.raises(ValueError, match="variational algorithm"):
            Blackjax("meanfield_vi")

    def test_smc_algorithm(self):
        with self.model, pytest.raises(ValueError, match="SMC requires"):
            Blackjax("tempered_smc")

    def test_sgmcmc_algorithm(self):
        with self.model, pytest.raises(ValueError, match="minibatch gradient"):
            Blackjax("sgld")

    def test_missing_required_parameter(self):
        with (
            self.model,
            pytest.raises(ValueError, match=r"requires explicit values for \['step_size'\]"),
        ):
            Blackjax("mala")

    def test_unknown_kwarg(self):
        with (
            self.model,
            pytest.raises(TypeError, match=r"Unknown keyword arguments \['stepsize'\]"),
        ):
            Blackjax("nuts", stepsize=0.1)

    def test_target_accept_unsupported(self):
        with self.model, pytest.raises(ValueError, match="target_accept is not supported"):
            Blackjax("mclmc", target_accept=0.9)

    def test_incompatible_adaptation(self):
        with self.model, pytest.raises(ValueError, match="adaptation='mclmc' can only be used"):
            Blackjax("nuts", adaptation="mclmc")

    def test_unknown_adaptation(self):
        with self.model, pytest.raises(ValueError, match="Unknown adaptation scheme"):
            Blackjax("nuts", adaptation="chees")

    def test_discrete_model(self):
        with pm.Model():
            pm.Poisson("k", 3)
            with pytest.raises(ValueError, match="can only sample models"):
                Blackjax()

    def test_tune_zero_with_adaptation(self):
        with self.model:
            sampler = Blackjax()
        with pytest.raises(ValueError, match="tune=0 is incompatible with adaptation"):
            sampler.sample(
                tune=0, draws=10, chains=1, initvals=None, random_seed=1, progressbar=False
            )


class TestSampleExternalSamplerArg:
    def test_clashes(self):
        with pm.Model() as model:
            pm.Normal("x", shape=3)
            sampler = Blackjax()

            with pytest.raises(ValueError, match="`step` and `external_sampler`"):
                pm.sample(external_sampler=sampler, step=pm.NUTS())
            with pytest.raises(ValueError, match="`nuts_sampler` and `external_sampler`"):
                pm.sample(external_sampler=sampler, nuts_sampler="blackjax")
            with pytest.raises(TypeError, match="Configure the sampler when constructing it"):
                pm.sample(external_sampler=sampler, target_accept=0.9)
            with pytest.raises(ValueError, match="return_inferencedata=False"):
                pm.sample(external_sampler=sampler, return_inferencedata=False)

    def test_model_mismatch(self):
        with pm.Model():
            pm.Normal("x", shape=3)
            sampler = Blackjax()
        with pm.Model():
            pm.Normal("y", shape=3)
            with pytest.raises(ValueError, match="does not match the model being sampled"):
                pm.sample(external_sampler=sampler)
