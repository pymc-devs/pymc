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

from pymc.sampling.samplers.base import Sampler, SamplerEntry
from pymc.sampling.samplers.step import StepSampler


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


RUN_KWARGS = {
    "chains": 2,
    "tune": 500,
    "draws": 300,
    "progressbar": False,
    "random_seed": 42,
    "compute_convergence_checks": False,
}


class TestStepSampler:
    def test_funnel(self, gaussian_model_data):
        model = make_model(gaussian_model_data)
        with model:
            idata = pm.sample(sampler=pm.StepSampler(nuts={"target_accept": 0.9}), **RUN_KWARGS)
        np.testing.assert_allclose(idata.posterior["mu"].mean(), 0.5, atol=0.2)

    def test_flat(self, gaussian_model_data):
        model = make_model(gaussian_model_data)
        idata = pm.StepSampler(init="adapt_diag").sample_from_init(model=model, **RUN_KWARGS)
        np.testing.assert_allclose(idata.posterior["mu"].mean(), 0.5, atol=0.2)

    def test_explicit_step(self, gaussian_model_data):
        model = make_model(gaussian_model_data)
        with model:
            sampler = pm.StepSampler(step=pm.Metropolis())
            idata = pm.sample(sampler=sampler, **RUN_KWARGS)
        assert idata.posterior.sizes["draw"] == 300

    def test_constructor_is_pure_configuration(self):
        sampler = pm.StepSampler(init="adapt_diag")  # no model context needed
        assert not hasattr(sampler, "model")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_matches_legacy_pm_sample(self, gaussian_model_data):
        """The funnel and the legacy path run the same engine with the same seed."""
        model = make_model(gaussian_model_data)
        with model:
            legacy = pm.sample(nuts_sampler="pymc", cores=1, **RUN_KWARGS)
            via_sampler = pm.sample(cores=1, sampler=pm.StepSampler(), **RUN_KWARGS)
        np.testing.assert_array_equal(
            legacy.posterior["mu"].values, via_sampler.posterior["mu"].values
        )


@pytest.mark.parametrize("library", ["nutpie", "numpyro", "blackjax"])
class TestExternalNUTSSamplers:
    @staticmethod
    def entry(library):
        pytest.importorskip(library)
        return getattr(pm, library).nuts

    def test_flat_entry_point(self, gaussian_model_data, library):
        model = make_model(gaussian_model_data)
        extra = {} if library == "nutpie" else {"chain_method": "vectorized"}
        idata = self.entry(library).sample(model=model, target_accept=0.9, **extra, **RUN_KWARGS)
        np.testing.assert_allclose(idata.posterior["mu"].mean(), 0.5, atol=0.2)
        assert idata.posterior.sizes == {"chain": 2, "draw": 300, "idx": 3}

    def test_funnel(self, gaussian_model_data, library):
        model = make_model(gaussian_model_data)
        extra = {} if library == "nutpie" else {"chain_method": "vectorized"}
        with model:
            idata = pm.sample(sampler=self.entry(library)(**extra), **RUN_KWARGS)
        np.testing.assert_allclose(idata.posterior["mu"].mean(), 0.5, atol=0.2)


class TestSamplerFunnel:
    """pm.sample(sampler=...) forwards run configuration and rejects the rest."""

    def _capture_run_kwargs(self, **sample_kwargs):
        with pm.Model():
            pm.Normal("x", shape=3)
            sampler = StepSampler()
            with mock.patch.object(StepSampler, "sample_from_init", return_value=None) as recorded:
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
        with pytest.raises(ValueError, match="`mp_ctx` is not supported"):
            self._capture_run_kwargs(mp_ctx="spawn")
        with pytest.raises(ValueError, match="`progressbar_theme` is not supported"):
            self._capture_run_kwargs(progressbar_theme="theme")

    def test_clashes(self):
        with pm.Model():
            pm.Normal("x", shape=3)
            sampler = StepSampler()

            with pytest.raises(ValueError, match="`step` and `sampler`"):
                pm.sample(sampler=sampler, step=pm.NUTS())
            with pytest.raises(ValueError, match="`nuts_sampler` and `sampler`"):
                pm.sample(sampler=sampler, nuts_sampler="pymc")
            with pytest.raises(TypeError, match="Configure the sampler when constructing it"):
                pm.sample(sampler=sampler, target_accept=0.9)
            with (
                pytest.warns(FutureWarning, match="return_inferencedata"),
                pytest.raises(ValueError, match="return_inferencedata=False"),
            ):
                pm.sample(sampler=sampler, return_inferencedata=False)

    def test_signatures_stay_in_sync(self):
        """The shared run configuration cannot drift between the surfaces."""
        run_params = dict(inspect.signature(Sampler.sample_from_init).parameters)
        run_params.pop("self")
        # every run parameter is a pm.sample parameter with the same name
        pm_sample_params = inspect.signature(pm.sample).parameters
        assert set(run_params) <= set(pm_sample_params)
        # the concrete samplers implement exactly the shared contract
        from pymc.sampling.samplers.jax_nuts import _JAXNUTS
        from pymc.sampling.samplers.nutpie import Nutpie

        for cls in (StepSampler, Nutpie, _JAXNUTS):
            params = dict(inspect.signature(cls.sample_from_init).parameters)
            params.pop("self")
            assert set(params) == set(run_params), cls
            assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in params.values())
        # the flat entry point is the run contract plus algorithm kwargs
        entry_params = dict(inspect.signature(SamplerEntry.sample).parameters)
        entry_params.pop("self")
        var_keyword = [
            name for name, p in entry_params.items() if p.kind is inspect.Parameter.VAR_KEYWORD
        ]
        assert var_keyword == ["algorithm_kwargs"]
        # `backend` is deliberate extra surface: the flat entry resolves it
        # into compile_kwargs like pm.sample does
        assert set(entry_params) - {"algorithm_kwargs", "backend"} == set(run_params)


class TestDeprecations:
    def test_nuts_sampler_deprecated(self, gaussian_model_data):
        pytest.importorskip("numpyro")
        model = make_model(gaussian_model_data)
        with model:
            with pytest.warns(DeprecationWarning, match="`nuts_sampler` argument is deprecated"):
                pm.sample(
                    nuts_sampler="numpyro",
                    chains=1,
                    tune=100,
                    draws=50,
                    progressbar=False,
                    random_seed=1,
                    compute_convergence_checks=False,
                )

    def test_nuts_init_arguments_deprecated(self, gaussian_model_data):
        model = make_model(gaussian_model_data)
        with model:
            with pytest.warns(DeprecationWarning, match="will move to the sampler configuration"):
                pm.sample(
                    init="adapt_diag",
                    chains=1,
                    tune=50,
                    draws=50,
                    progressbar=False,
                    random_seed=1,
                    compute_convergence_checks=False,
                )

    def test_no_warning_by_default(self, gaussian_model_data):
        import warnings as _warnings

        model = make_model(gaussian_model_data)
        with model:
            with _warnings.catch_warnings():
                _warnings.simplefilter("error", DeprecationWarning)
                pm.sample(
                    chains=1,
                    tune=50,
                    draws=50,
                    progressbar=False,
                    random_seed=1,
                    compute_convergence_checks=False,
                )


class TestAPIEquivalenceRegressions:
    """Replacements must accept what the deprecated pm.sample arguments accepted."""

    def test_array_like_random_seed(self):
        pytest.importorskip("numpyro")
        with pm.Model() as model:
            pm.Normal("x", shape=3)
        captured = {}
        with mock.patch(
            "pymc.sampling.jax.sample_jax_nuts", side_effect=lambda **kw: captured.update(kw)
        ):
            pm.numpyro.nuts().sample_from_init(model=model, chains=2, random_seed=[1, 2])
        assert isinstance(captured["random_seed"], int)

        from pymc.sampling.samplers.nutpie import Nutpie

        pytest.importorskip("nutpie")
        captured.clear()
        with mock.patch(
            "pymc.sampling.mcmc._sample_external_nuts",
            side_effect=lambda **kw: captured.update(kw),
        ):
            Nutpie().sample_from_init(model=model, chains=2, random_seed=[1, 2])
        assert isinstance(captured["random_seed"][0], int)

    def test_tune_none_reaches_step_defaults(self):
        with pm.Model() as model:
            pm.Normal("x", shape=3)
        captured = {}
        with mock.patch(
            "pymc.sampling.mcmc._sample_with_step_methods",
            side_effect=lambda **kw: captured.update(kw),
        ):
            StepSampler().sample_from_init(model=model, tune=None)
            assert captured["tune"] is None
            with model:
                pm.sample(sampler=StepSampler())  # pm.sample's tune default is None
            assert captured["tune"] is None

    def test_flat_entry_resolves_backend(self):
        pytest.importorskip("numpyro")
        from pymc.sampling.samplers.jax_nuts import NumpyroNUTS

        with pm.Model() as model:
            pm.Normal("x", shape=3)
        with mock.patch.object(NumpyroNUTS, "sample_from_init", return_value=None) as recorded:
            pm.numpyro.nuts.sample(model=model, backend="jax", target_accept=0.9)
        compile_kwargs = recorded.call_args.kwargs["compile_kwargs"]
        assert "mode" in compile_kwargs

    def test_nutpie_quiet_disables_progressbar(self):
        pytest.importorskip("nutpie")
        from pymc.sampling.samplers.nutpie import Nutpie

        with pm.Model() as model:
            pm.Normal("x", shape=3)
        captured = {}
        with mock.patch(
            "pymc.sampling.mcmc._sample_external_nuts",
            side_effect=lambda **kw: captured.update(kw),
        ):
            Nutpie().sample_from_init(model=model, quiet=True)
        assert captured["progressbar"] is False


class TestExternalSamplerDependencies:
    def test_missing_package_raises_with_hint(self):
        from pymc.sampling.samplers.base import ExternalSampler

        class Fake(ExternalSampler):
            package = "definitely_not_installed_xyz"

            def sample_from_init(self, **kwargs):
                pass

        with pytest.raises(ImportError, match="pip install definitely_not_installed_xyz"):
            Fake()
