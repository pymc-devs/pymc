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
import warnings

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytensor.tensor as pt
import pytest
import xarray as xr

from pymc import (
    NUTS,
    Data,
    Deterministic,
    HalfNormal,
    Metropolis,
    Model,
    Normal,
    Potential,
    sample,
)
from pymc.exceptions import SamplingError
from pymc.progress_bar import NutpieProgressBarManager

pytestmark = pytest.mark.filterwarnings(
    "error",
    "ignore:There are not enough devices to run parallel chains:UserWarning",
    "ignore:os.fork\\(\\) was called:RuntimeWarning",
)


@pytest.mark.parametrize("nuts_sampler", ["pymc", "nutpie", "blackjax", "numpyro"])
def test_external_nuts_sampler(nuts_sampler):
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


@pytest.mark.parametrize("nuts_sampler", ["pymc", "nutpie", "blackjax", "numpyro"])
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


@pytest.mark.parametrize(
    "initvals",
    [
        None,
        {"x1": 10000.0, "x2": 0.0},
        {"x1": 10000.0},
        {"x1_log__": np.log(10000.0)},
    ],
    ids=["negative_control", "full_constrained", "partial_constrained", "unconstrained_partial"],
)
@pytest.mark.parametrize("nuts_sampler", ["pymc", "nutpie", "blackjax", "numpyro"])
def test_initvals(nuts_sampler, initvals):
    if nuts_sampler != "pymc":
        pytest.importorskip(nuts_sampler)

    # `x1` has a log transform; we pin it to 10000. Three guards check the value
    # is mapped through the transform correctly:
    #   - the `x1 - x2 > 1000` constraint excludes log(10000) ≈ 9.2 (wrong:
    #     `x1_log__` value taken as constrained), and excludes default init
    #     (x1 ≈ 79.8 ± jitter < 1000),
    #   - the `x1 < 1e6` upper bound excludes exp(10000) (wrong: dict value
    #     taken as if it were already on the transformed/unconstrained space).
    with Model():
        x1 = HalfNormal("x1", 100)
        x2 = Normal("x2", 0, 1)
        Potential("c", pt.where(x1 - x2 > 1000, 0.0, -np.inf))

        with pytest.raises((SamplingError, RuntimeError)) if initvals is None else nullcontext():
            idata = sample(
                nuts_sampler=nuts_sampler,
                tune=1,
                draws=3,
                chains=2,
                progressbar=False,
                random_seed=0,
                compute_convergence_checks=False,
                initvals=initvals,
            )
        if initvals is None:
            return

    assert idata.posterior.chain.size == 2
    assert ((idata.posterior["x1"] - idata.posterior["x2"]) > 1000).all()
    assert (idata.posterior["x1"] < 1e6).all()


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


@pytest.fixture
def patched_sampler():
    """Pretend nutpie is installed and intercept any dispatch to it."""
    with (
        mock.patch("pymc.sampling.mcmc.NUTPIE_INSTALLED", True),
        mock.patch("pymc.sampling.mcmc._sample_external_nuts") as mock_ext,
    ):
        yield mock_ext


class TestExternalSamplerKwargCompat:
    """Validate how `pm.sample` handles kwargs that external samplers don't fully honor."""

    with Model() as _model:
        Normal("x")

    _BASE_KWARGS = {
        "tune": 2,
        "draws": 2,
        "chains": 1,
        "progressbar": False,
        "compile_kwargs": {"mode": "NUMBA"},
    }

    @pytest.mark.parametrize(
        "extra",
        [
            {"return_inferencedata": False},
            {"trace": object()},
            {"callback": lambda **kw: None},
        ],
        ids=["return_inferencedata_false", "custom_trace", "callback"],
    )
    def test_explicit_nutpie_raises_on_incompatible(self, patched_sampler, extra):
        pytest.importorskip("nutpie")

        # Filter the separate FutureWarning for return_inferencedata=False; we only
        # care that pm.sample raises the external-sampler ValueError.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*return_inferencedata=False.*", FutureWarning)
            with self._model:
                with pytest.raises(ValueError, match="`nuts_sampler='nutpie'`"):
                    sample(nuts_sampler="nutpie", **self._BASE_KWARGS, **extra)
        patched_sampler.assert_not_called()

    def test_explicit_nutpie_raises_on_non_nuts_step(self, patched_sampler):
        pytest.importorskip("nutpie")
        with self._model:
            step = Metropolis()
            with pytest.raises(ValueError, match="not assigned to another step sampler"):
                sample(nuts_sampler="nutpie", step=step, **self._BASE_KWARGS)
        patched_sampler.assert_not_called()

    def test_explicit_nutpie_warns_on_non_default_init(self, patched_sampler):
        pytest.importorskip("nutpie")
        with self._model:
            with pytest.warns(UserWarning, match="`init='advi'` is ignored"):
                sample(nuts_sampler="nutpie", init="advi", **self._BASE_KWARGS)
        patched_sampler.assert_called_once()

    def test_explicit_nutpie_warns_on_provided_nuts_step(self, patched_sampler):
        pytest.importorskip("nutpie")
        with self._model:
            step = NUTS()
            with pytest.warns(UserWarning, match="NUTS `step` is ignored"):
                sample(nuts_sampler="nutpie", step=step, **self._BASE_KWARGS)
        patched_sampler.assert_called_once()

    def test_explicit_nutpie_raises_on_per_chain_initvals(self):
        pytest.importorskip("nutpie")
        with self._model, pytest.raises(NotImplementedError, match="per-chain"):
            sample(
                nuts_sampler="nutpie",
                initvals=[{"x": 0.0}, {"x": 1.0}],
                tune=1,
                draws=1,
                chains=2,
                progressbar=False,
            )


class TestNutpieAutoSelection:
    """Validate when `pm.sample` auto-selects nutpie based on env/compile mode."""

    with Model() as _model:
        Normal("x")

    _BASE_KWARGS = {
        "tune": 10,
        "draws": 10,
        "chains": 1,
        "progressbar": False,
    }

    @pytest.mark.parametrize("mode", ["NUMBA", "JAX"])
    @pytest.mark.parametrize("via", ["compile_kwargs", "backend"])
    def test_auto_selects_nutpie_when_installed(self, patched_sampler, mode, via):
        extra = {"compile_kwargs": {"mode": mode}} if via == "compile_kwargs" else {"backend": mode}
        with self._model:
            sample(**self._BASE_KWARGS, **extra)
        assert patched_sampler.call_args.kwargs["sampler"] == "nutpie"

    def test_falls_back_to_pymc_when_nutpie_missing(self):
        with (
            mock.patch("pymc.sampling.mcmc.NUTPIE_INSTALLED", False),
            mock.patch("pymc.sampling.mcmc._sample_external_nuts") as mock_ext,
        ):
            with self._model:
                sample(**self._BASE_KWARGS)
        mock_ext.assert_not_called()

    @pytest.mark.parametrize(
        "arg_name,arg",
        [
            ("return_inferencedata", False),
            ("trace", object()),
            ("callback", lambda **kw: None),
            ("init", "advi"),
            ("step", Metropolis),
            # Non numba/jax backends
            ("backend", "c"),
            ("compile_kwargs", {"mode": "FAST_COMPILE"}),
            # Per-chain initvals
            ("initvals", [{"x": 0.0}, {"x": 1.0}]),
        ],
    )
    def test_falls_back_to_pymc_for_disqualifying_kwargs(self, patched_sampler, arg_name, arg):
        # Each of these kwargs disqualifies nutpie auto-selection; pm.sample should
        # route to the pymc sampler (external stub never called) instead of raising.
        # Numba is the default linker, so we don't need to set `compile_kwargs` to
        # qualify for nutpie auto-pick — that means each entry can stand alone here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*return_inferencedata=False.*", FutureWarning)
            with self._model:
                try:
                    sample(**self._BASE_KWARGS, **{arg_name: arg})
                except Exception:
                    # The pymc path may error on the stand-in objects (dummy trace, etc.);
                    # we only care that nutpie wasn't dispatched.
                    pass
        patched_sampler.assert_not_called()

    def test_falls_back_to_pymc_for_configured_nuts_step(self, patched_sampler):
        # A pre-built NUTS instance with a non-default argument (here a custom
        # mass-matrix potential) carries user state that nutpie cannot consume.
        # Auto-selection must fall back to the pymc sampler rather than silently
        # dropping the configuration.
        from pymc.step_methods.hmc.quadpotential import QuadPotentialDiag

        with self._model:
            step = NUTS(potential=QuadPotentialDiag(np.ones(1)))
            sample(step=step, **self._BASE_KWARGS)
        patched_sampler.assert_not_called()
