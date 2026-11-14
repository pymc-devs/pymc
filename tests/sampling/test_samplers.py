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

import inspect
import unittest.mock as mock

import numpy as np
import pytest

import pymc as pm


# Importing jax/blackjax initializes the jax backend, which must not happen at
# pytest collection time: it would run before pymc.sampling.jax can set
# XLA_FLAGS (breaking pmap parallel chains in other test modules) and has been
# observed to destabilize XLA compilation in unrelated tests. Both are instead
# imported lazily via the `blackjax` fixture below.
# `Blackjax` itself is safe to import: it defers all jax/blackjax imports.
from pymc.sampling.samplers.base import Sampler
from pymc.sampling.samplers.blackjax import AlgorithmEntry, Blackjax


@pytest.fixture(scope="module", autouse=True)
def blackjax():
    pytest.importorskip("jax")
    # Sets XLA_FLAGS before the jax backend initializes
    import pymc.sampling.jax  # noqa: F401

    return pytest.importorskip("blackjax")


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
    # The flat entry point: algorithm and run arguments in one call
    idata = getattr(pm.blackjax, algorithm).sample(
        model=model,
        chain_method="vectorized",
        chains=2,
        tune=700,
        draws=500,
        progressbar=False,
        random_seed=42,
        compute_convergence_checks=False,
        **kwargs,
    )

    assert idata.posterior.sizes == {"chain": 2, "draw": 500, "idx": 3}
    assert "lp" in idata.sample_stats
    # Gradient-based tuned algorithms should recover the true parameters
    if algorithm in ("nuts", "hmc", "mclmc"):
        np.testing.assert_allclose(idata.posterior["mu"].mean(), 0.5, atol=0.2)
        np.testing.assert_allclose(idata.posterior["sigma"].mean(), 1.2, atol=0.2)
        if algorithm != "mclmc":
            assert "diverging" in idata.sample_stats


def test_sample_funnel(gaussian_model_data):
    """pm.sample(sampler=...) is sugar over sample_from_init."""
    model = make_model(gaussian_model_data)
    with model:
        idata = pm.sample(
            sampler=pm.blackjax.nuts(target_accept=0.9, chain_method="vectorized"),
            chains=2,
            tune=500,
            draws=300,
            progressbar=False,
            random_seed=42,
            compute_convergence_checks=False,
        )
    np.testing.assert_allclose(idata.posterior["mu"].mean(), 0.5, atol=0.2)


def test_blackjax_algorithm_object(blackjax):
    sampler = Blackjax(blackjax.nuts, chain_method="vectorized")
    assert sampler.algorithm_name == "nuts"
    assert sampler.adaptation == "window"


def test_blackjax_stat_renames(gaussian_model_data):
    model = make_model(gaussian_model_data)
    idata = pm.blackjax.nuts.sample(
        model=model,
        chain_method="vectorized",
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
    kwargs = Blackjax(target_accept=0.95).get_kwargs()
    assert kwargs["algorithm"]["step_size"] == "<required>"
    assert kwargs["algorithm"]["max_num_doublings"] == 10
    assert kwargs["adaptation"]["target_acceptance_rate"] == 0.95


def test_blackjax_kwargs_routing():
    sampler = Blackjax(
        "nuts", max_num_doublings=8, is_mass_matrix_diagonal=False, target_accept=0.9
    )
    assert sampler.algorithm_kwargs == {"max_num_doublings": 8}
    assert sampler.adaptation_kwargs == {
        "is_mass_matrix_diagonal": False,
        "target_acceptance_rate": 0.9,
    }


def test_constructor_is_pure_configuration():
    """The constructor is algorithm configuration only: no model binding."""
    sampler = Blackjax()  # no model context needed
    assert not hasattr(sampler, "model")
    # the model arrives at sample time, from the context if not passed
    with pm.Model():
        pm.Poisson("k", 3)
        with pytest.raises(ValueError, match="can only sample models"):
            sampler.sample_from_init(draws=10, tune=10)


class TestBlackjaxErrors:
    def test_unknown_algorithm(self):
        with pytest.raises(ValueError, match="Unknown blackjax algorithm 'nutz'"):
            Blackjax("nutz")

    def test_vi_algorithm(self):
        with pytest.raises(ValueError, match="variational algorithm"):
            Blackjax("meanfield_vi")

    def test_smc_algorithm(self):
        with pytest.raises(ValueError, match="SMC requires"):
            Blackjax("tempered_smc")

    def test_sgmcmc_algorithm(self):
        with pytest.raises(ValueError, match="minibatch gradient"):
            Blackjax("sgld")

    def test_missing_required_parameter(self):
        with pytest.raises(ValueError, match=r"requires explicit values for \['step_size'\]"):
            Blackjax("mala")

    def test_unknown_kwarg(self):
        with pytest.raises(TypeError, match=r"Unknown keyword arguments \['stepsize'\]"):
            Blackjax("nuts", stepsize=0.1)

    def test_target_accept_unsupported(self):
        with pytest.raises(ValueError, match="target_accept is not supported"):
            Blackjax("mclmc", target_accept=0.9)

    def test_incompatible_adaptation(self):
        with pytest.raises(ValueError, match="adaptation='mclmc' can only be used"):
            Blackjax("nuts", adaptation="mclmc")

    def test_unknown_adaptation(self):
        with pytest.raises(ValueError, match="Unknown adaptation scheme"):
            Blackjax("nuts", adaptation="chees")

    def test_tune_zero_with_adaptation(self):
        sampler = Blackjax()
        with pm.Model():
            pm.Normal("x", shape=3)
            with pytest.raises(ValueError, match="tune=0 is incompatible with adaptation"):
                sampler.sample_from_init(tune=0, draws=10, progressbar=False)


class TestAlgorithmNamespace:
    def test_entry_surfaces(self):
        entry = pm.blackjax.mclmc
        assert isinstance(entry, AlgorithmEntry)
        assert "blackjax.mclmc" in entry.__doc__
        # calling configures a sampler for the funnel
        sampler = entry()
        assert isinstance(sampler, Blackjax)
        assert sampler.algorithm_name == "mclmc"
        assert sampler.adaptation == "mclmc"
        # .sample is the flat entry point
        assert callable(entry.sample)

    def test_entry_kwargs(self):
        sampler = pm.blackjax.nuts(target_accept=0.9, max_num_doublings=8)
        assert sampler.algorithm_kwargs == {"max_num_doublings": 8}

    def test_dir_lists_algorithms(self):
        names = dir(pm.blackjax)
        assert {"nuts", "hmc", "mclmc", "mala", "barker", "Blackjax"} <= set(names)
        assert "sgld" not in names

    def test_unsupported_attribute(self):
        with pytest.raises(AttributeError, match="minibatch gradient"):
            pm.blackjax.sgld
        with pytest.raises(AttributeError, match="variational algorithm"):
            pm.blackjax.meanfield_vi
        with pytest.raises(AttributeError, match="no attribute 'nutz'"):
            pm.blackjax.nutz


class TestSamplerFunnel:
    """pm.sample(sampler=...) forwards run configuration and rejects the rest."""

    def _capture_run_kwargs(self, **sample_kwargs):
        with pm.Model():
            pm.Normal("x", shape=3)
            sampler = Blackjax()
            with mock.patch.object(Blackjax, "sample_from_init", return_value=None) as recorded:
                pm.sample(sampler=sampler, **sample_kwargs)
        return recorded.call_args.kwargs

    def test_run_configuration_forwarded(self):
        kwargs = self._capture_run_kwargs(draws=11, tune=22, discard_tuned_samples=False)
        assert kwargs["draws"] == 11
        assert kwargs["tune"] == 22
        assert kwargs["discard_tuned_samples"] is False
        assert kwargs["compile_kwargs"] == {}
        assert kwargs["model"] is not None

    def test_nuts_specific_arguments_rejected(self):
        with pytest.raises(ValueError, match="`init` is NUTS-specific"):
            self._capture_run_kwargs(init="adapt_diag")
        with pytest.raises(ValueError, match="`n_init` is NUTS-specific"):
            self._capture_run_kwargs(n_init=1000)
        with pytest.raises(ValueError, match="`jitter_max_retries` is not supported"):
            self._capture_run_kwargs(jitter_max_retries=3)

    def test_clashes(self):
        with pm.Model():
            pm.Normal("x", shape=3)
            sampler = Blackjax()

            with pytest.raises(ValueError, match="`step` and `sampler`"):
                pm.sample(sampler=sampler, step=pm.NUTS())
            with pytest.raises(ValueError, match="`nuts_sampler` and `sampler`"):
                pm.sample(sampler=sampler, nuts_sampler="blackjax")
            with pytest.raises(TypeError, match="Configure the sampler when constructing it"):
                pm.sample(sampler=sampler, target_accept=0.9)
            with pytest.raises(ValueError, match="return_inferencedata=False"):
                pm.sample(sampler=sampler, return_inferencedata=False)

    def test_signatures_stay_in_sync(self):
        """The shared run configuration cannot drift between the surfaces."""
        run_params = dict(inspect.signature(Sampler.sample_from_init).parameters)
        run_params.pop("self")
        # every run parameter is a pm.sample parameter with the same name
        pm_sample_params = inspect.signature(pm.sample).parameters
        assert set(run_params) <= set(pm_sample_params)
        # the concrete sampler implements exactly the shared contract
        blackjax_params = dict(inspect.signature(Blackjax.sample_from_init).parameters)
        blackjax_params.pop("self")
        assert set(blackjax_params) == set(run_params)
        assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in blackjax_params.values())
        # the flat entry point is the run contract plus algorithm kwargs
        entry_params = dict(inspect.signature(AlgorithmEntry.sample).parameters)
        entry_params.pop("self")
        var_keyword = [
            name for name, p in entry_params.items() if p.kind is inspect.Parameter.VAR_KEYWORD
        ]
        assert var_keyword == ["algorithm_kwargs"]
        assert set(entry_params) - {"algorithm_kwargs"} == set(run_params)


class TestBlackjaxArgumentDispositions:
    """Each run argument has an explicit, argued disposition."""

    def test_compile_kwargs_non_jax_mode_raises(self):
        sampler = Blackjax()
        with pytest.raises(ValueError, match="always compiles the model with the jax backend"):
            sampler._validate_compile_kwargs({"mode": "NUMBA"})
        with pytest.raises(ValueError, match=r"\['random_state'\] are not supported"):
            sampler._validate_compile_kwargs({"mode": "JAX", "random_state": 1})
        # jax mode and no request at all are fine
        sampler._validate_compile_kwargs({"mode": "JAX"})
        sampler._validate_compile_kwargs(None)
        sampler._validate_compile_kwargs({})

    def test_backend_through_pm_sample_raises(self):
        with pm.Model():
            pm.Normal("x", shape=3)
            sampler = Blackjax()
            with pytest.raises(ValueError, match="always compiles the model with the jax backend"):
                pm.sample(sampler=sampler, backend="numba")

    @pytest.mark.parametrize(
        "sample_kwargs,warning_match",
        [
            ({"discard_tuned_samples": False}, "discards warmup draws"),
            ({"keep_warning_stat": True}, "does not emit"),
        ],
    )
    def test_unsupported_run_options_warn(self, sample_kwargs, warning_match):
        sampler = Blackjax()
        with pm.Model():
            pm.Normal("x", shape=3)
            # Abort right after the argument dispositions ran
            with (
                mock.patch(
                    "pymc.sampling.samplers.blackjax._get_seeds_per_chain",
                    side_effect=RuntimeError("stop"),
                ),
                pytest.warns(UserWarning, match=warning_match),
                pytest.raises(RuntimeError, match="stop"),
            ):
                sampler.sample_from_init(progressbar=False, **sample_kwargs)
