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
import logging
import unittest.mock as mock
import warnings

from contextlib import ExitStack as does_not_raise

import numpy as np
import numpy.testing as npt
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.special

from arviz import InferenceData
from pytensor import shared
from pytensor.compile.ops import as_op

import pymc as pm

from pymc.backends.ndarray import NDArray
from pymc.distributions import transforms
from pymc.exceptions import SamplingError
from pymc.sampling.mcmc import assign_step_methods
from pymc.stats.convergence import SamplerWarning, WarningType
from pymc.step_methods import (
    NUTS,
    BinaryGibbsMetropolis,
    CategoricalGibbsMetropolis,
    CompoundStep,
    HamiltonianMC,
    Metropolis,
    Slice,
)
from pymc.testing import fast_unstable_sampling_mode
from tests.models import simple_init


class TestInitNuts:
    def setup_method(self):
        self.model, self.start, self.step, _ = simple_init()

    def test_checks_seeds_kwarg(self):
        with self.model:
            with pytest.raises(ValueError, match="Number of seeds"):
                pm.sampling.mcmc.init_nuts(chains=2, random_seed=[1])


class TestSample:
    def setup_method(self):
        self.model, self.start, self.step, _ = simple_init()

    @pytest.mark.parametrize("init", ("jitter+adapt_diag", "advi", "map"))
    @pytest.mark.parametrize("cores", (1, 2))
    @pytest.mark.parametrize(
        "chains, seeds",
        [
            (1, None),
            (1, 1),
            (1, [1]),
            (2, None),
            (2, 1),
            (2, [1, 2]),
        ],
    )
    def test_random_seed(self, chains, seeds, cores, init):
        with pm.Model():
            x = pm.Normal("x", 0, 10, initval="prior")
            tr1 = pm.sample(
                chains=chains,
                random_seed=seeds,
                cores=cores,
                init=init,
                tune=0,
                draws=10,
                return_inferencedata=False,
                compute_convergence_checks=False,
            )
            tr2 = pm.sample(
                chains=chains,
                random_seed=seeds,
                cores=cores,
                init=init,
                tune=0,
                draws=10,
                return_inferencedata=False,
                compute_convergence_checks=False,
            )

        allequal = np.all(tr1["x"] == tr2["x"])
        if seeds is None:
            assert not allequal
        else:
            assert allequal

    @mock.patch("numpy.random.seed")
    def test_default_sample_does_not_set_global_seed(self, mocked_seed):
        # Test that when random_seed is None, `np.random.seed` is not called in the main
        # process. Ideally it would never be called, but PyMC step samplers still rely
        # on global seeding for reproducible behavior.
        kwargs = dict(tune=2, draws=2, random_seed=None)
        with self.model:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                pm.sample(chains=1, **kwargs)
                pm.sample(chains=2, cores=1, **kwargs)
                pm.sample(chains=2, cores=2, **kwargs)
        mocked_seed.assert_not_called()

    def test_sample_does_not_rely_on_external_global_seeding(self):
        # Tests that sampling does not depend on exertenal global seeding
        kwargs = dict(
            tune=2,
            draws=20,
            random_seed=None,
            return_inferencedata=False,
        )
        with self.model:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                np.random.seed(1)
                idata11 = pm.sample(chains=1, **kwargs)
                np.random.seed(1)
                idata12 = pm.sample(chains=2, cores=1, **kwargs)
                np.random.seed(1)
                idata13 = pm.sample(chains=2, cores=2, **kwargs)

                np.random.seed(1)
                idata21 = pm.sample(chains=1, **kwargs)
                np.random.seed(1)
                idata22 = pm.sample(chains=2, cores=1, **kwargs)
                np.random.seed(1)
                idata23 = pm.sample(chains=2, cores=2, **kwargs)

        assert np.all(idata11["x"] != idata21["x"])
        assert np.all(idata12["x"] != idata22["x"])
        assert np.all(idata13["x"] != idata23["x"])

    def test_sample_init(self):
        with self.model:
            for init in (
                "advi",
                "advi_map",
                "map",
                "adapt_diag",
                "jitter+adapt_diag",
                "jitter+adapt_diag_grad",
                "adapt_full",
                "jitter+adapt_full",
            ):
                kwargs = {
                    "init": init,
                    "tune": 120,
                    "n_init": 1000,
                    "draws": 50,
                    "random_seed": 20160911,
                }
                with warnings.catch_warnings(record=True) as rec:
                    warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                    if init.endswith("adapt_full"):
                        with pytest.warns(UserWarning, match="experimental feature"):
                            pm.sample(**kwargs)
                    else:
                        pm.sample(**kwargs)

    def test_sample_args(self):
        with self.model:
            with pytest.raises(ValueError, match=r"'foo'"):
                pm.sample(50, tune=0, chains=1, step=pm.Metropolis(), foo=1)

            with pytest.raises(ValueError, match=r"'foo'") as excinfo:
                pm.sample(50, tune=0, chains=1, step=pm.Metropolis(), foo={})

    def test_parallel_start(self):
        with self.model:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                idata = pm.sample(
                    0,
                    tune=5,
                    cores=2,
                    discard_tuned_samples=False,
                    initvals=[{"x": [10, 10]}, {"x": [-10, -10]}],
                    random_seed=20160911,
                )
        assert idata.warmup_posterior["x"].sel(chain=0, draw=0).values[0] > 0
        assert idata.warmup_posterior["x"].sel(chain=1, draw=0).values[0] < 0

    def test_reset_tuning(self):
        with self.model:
            tune = 50
            chains = 2
            start, step = pm.sampling.mcmc.init_nuts(chains=chains, random_seed=[1, 2])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                pm.sample(draws=2, tune=tune, chains=chains, step=step, initvals=start, cores=1)
            assert step.potential._n_samples == tune
            assert step.step_adapt._count == tune + 1

    @pytest.mark.parametrize(
        "start, error",
        [
            ({"x": 1}, ValueError),
            ({"x": [1, 2, 3]}, ValueError),
            ({"x": np.array([[1, 1], [1, 1]])}, ValueError),
        ],
    )
    def test_sample_start_bad_shape(self, start, error):
        with pytest.raises(error):
            pm.sampling.mcmc._check_start_shape(self.model, start)

    @pytest.mark.parametrize("start", [{"x": np.array([1, 1])}, {"x": [10, 10]}, {"x": [-10, -10]}])
    def test_sample_start_good_shape(self, start):
        pm.sampling.mcmc._check_start_shape(self.model, start)

    def test_sample_callback(self):
        callback = mock.Mock()
        test_cores = [1, 2]
        with self.model:
            for cores in test_cores:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                    pm.sample(
                        10,
                        tune=0,
                        chains=2,
                        step=self.step,
                        cores=cores,
                        random_seed=20160911,
                        callback=callback,
                    )
                assert callback.called

    def test_callback_can_cancel(self):
        trace_cancel_length = 5

        def callback(trace, draw):
            if len(trace) >= trace_cancel_length:
                raise KeyboardInterrupt()

        with self.model:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                trace = pm.sample(
                    10,
                    tune=0,
                    chains=1,
                    step=self.step,
                    cores=1,
                    random_seed=2016911,
                    callback=callback,
                    return_inferencedata=False,
                )
            assert len(trace) == trace_cancel_length

    def test_sequential_backend(self):
        with self.model:
            backend = NDArray()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                pm.sample(10, tune=5, cores=1, chains=2, step=pm.Metropolis(), trace=backend)

    @pytest.mark.parametrize("symbolic_rv", (False, True))
    def test_deterministic_of_unobserved(self, symbolic_rv):
        with pm.Model() as model:
            if symbolic_rv:
                x = pm.Censored(
                    "x", pm.HalfNormal.dist(1), lower=None, upper=10, transform=transforms.log
                )
            else:
                x = pm.HalfNormal("x", 1)
            y = pm.Deterministic("y", x + 100)
            idata = pm.sample(
                chains=1,
                tune=10,
                draws=50,
                compute_convergence_checks=False,
            )

        np.testing.assert_allclose(idata.posterior["y"], idata.posterior["x"] + 100)

    @pytest.mark.parametrize("symbolic_rv", (False, True))
    def test_transform_with_rv_dependency(self, symbolic_rv):
        # Test that untransformed variables that depend on upstream variables are properly handled
        with pm.Model() as m:
            if symbolic_rv:
                x = pm.Censored("x", pm.HalfNormal.dist(1), lower=0, upper=1, observed=1)
            else:
                x = pm.HalfNormal("x", observed=1)

            transform = pm.distributions.transforms.Interval(
                bounds_fn=lambda *inputs: (inputs[-2], inputs[-1])
            )
            y = pm.Uniform("y", lower=0, upper=x, transform=transform)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                trace = pm.sample(tune=10, draws=50, return_inferencedata=False, random_seed=336)

        assert np.allclose(scipy.special.expit(trace["y_interval__"]), trace["y"])


class ApocalypticMetropolis(pm.Metropolis):
    """A stepper that warns in every iteration."""

    stats_dtypes_shapes = {
        **pm.Metropolis.stats_dtypes_shapes,
        "warning": (SamplerWarning, None),
    }

    def astep(self, q0):
        draw, stats = super().astep(q0)
        stats[0]["warning"] = SamplerWarning(
            WarningType.BAD_ENERGY,
            "Asteroid incoming!",
            "warn",
        )
        return draw, stats


class TestSampleReturn:
    """Tests related to kwargs that parametrize how `pm.sample` results are returned."""

    def test_sample_return_lengths(self):
        with pm.Model() as model:
            pm.Normal("n")

            # Get a MultiTrace with warmup
            with pytest.warns(UserWarning, match="will be included"):
                mtrace = pm.sample(
                    draws=100,
                    tune=50,
                    cores=1,
                    chains=3,
                    step=pm.Metropolis(),
                    return_inferencedata=False,
                    discard_tuned_samples=False,
                )
                assert isinstance(mtrace, pm.backends.base.MultiTrace)
                assert len(mtrace) == 150

        # Now instead of running more MCMCs, we'll test the other return
        # options using the basetraces inside the MultiTrace.
        traces = list(mtrace._straces.values())
        assert len(traces) == 3

        # MultiTrace without warmup
        mtrace_pst = pm.sampling.mcmc._sample_return(
            run=None,
            traces=traces,
            tune=50,
            t_sampling=123.4,
            discard_tuned_samples=True,
            return_inferencedata=False,
            compute_convergence_checks=False,
            keep_warning_stat=True,
            idata_kwargs={},
            model=model,
        )
        assert isinstance(mtrace_pst, pm.backends.base.MultiTrace)
        assert len(mtrace_pst) == 100
        assert mtrace_pst.report.t_sampling == 123.4
        assert mtrace_pst.report.n_tune == 50
        assert mtrace_pst.report.n_draws == 100

        # InferenceData with warmup
        idata_w = pm.sampling.mcmc._sample_return(
            run=None,
            traces=traces,
            tune=50,
            t_sampling=123.4,
            discard_tuned_samples=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
            keep_warning_stat=True,
            idata_kwargs={},
            model=model,
        )
        assert isinstance(idata_w, InferenceData)
        assert hasattr(idata_w, "warmup_posterior")
        assert idata_w.warmup_posterior.sizes["draw"] == 50
        assert idata_w.posterior.sizes["draw"] == 100
        assert idata_w.posterior.sizes["chain"] == 3

        # InferenceData without warmup
        idata = pm.sampling.mcmc._sample_return(
            run=None,
            traces=traces,
            tune=50,
            t_sampling=123.4,
            discard_tuned_samples=True,
            compute_convergence_checks=False,
            return_inferencedata=True,
            keep_warning_stat=False,
            idata_kwargs={},
            model=model,
        )
        assert isinstance(idata, InferenceData)
        assert not hasattr(idata, "warmup_posterior")
        assert idata.posterior.sizes["draw"] == 100
        assert idata.posterior.sizes["chain"] == 3

    @pytest.mark.parametrize("cores", [1, 2])
    def test_logs_sampler_warnings(self, caplog, cores):
        """Asserts that "warning" sampler stats are logged during sampling."""
        with pm.Model():
            pm.Normal("n")
            with caplog.at_level(logging.WARNING):
                idata = pm.sample(
                    tune=2,
                    draws=3,
                    cores=cores,
                    chains=cores,
                    step=ApocalypticMetropolis(),
                    compute_convergence_checks=False,
                    discard_tuned_samples=False,
                    keep_warning_stat=True,
                )

        # Sampler warnings should be logged
        nwarns = sum("Asteroid" in rec.message for rec in caplog.records)
        assert nwarns == (2 + 3) * cores

    @pytest.mark.parametrize("keep_warning_stat", [None, True])
    def test_keep_warning_stat_setting(self, keep_warning_stat):
        """The ``keep_warning_stat`` stat (aka "Adrian's kwarg) enables users
        to keep the ``SamplerWarning`` objects from the ``sample_stats.warning`` group.
        This breaks ``idata.to_netcdf()`` which is why it defaults to ``False``.
        """
        sample_kwargs = dict(
            tune=2,
            draws=3,
            chains=1,
            compute_convergence_checks=False,
            discard_tuned_samples=False,
            keep_warning_stat=keep_warning_stat,
        )
        if keep_warning_stat:
            sample_kwargs["keep_warning_stat"] = True
        with pm.Model():
            pm.Normal("n")
            idata = pm.sample(step=ApocalypticMetropolis(), **sample_kwargs)

        if keep_warning_stat:
            assert "warning" in idata.warmup_sample_stats
            assert "warning" in idata.sample_stats
            # And end up in the InferenceData
            assert "warning" in idata.sample_stats
            # NOTE: The stats are squeezed by default but this does not always work.
            #       This tests flattens so we don't have to be exact in accessing (non-)squeezed items.
            #       Also see https://github.com/pymc-devs/pymc/issues/6207.
            warn_objs = list(idata.sample_stats.warning.sel(chain=0).values.flatten())
            assert warn_objs
            if isinstance(warn_objs[0], np.ndarray):
                # Squeeze warning stats. See https://github.com/pymc-devs/pymc/issues/6207
                warn_objs = [a.tolist() for a in warn_objs]
            assert any(isinstance(w, SamplerWarning) for w in warn_objs)
            assert any("Asteroid" in w.message for w in warn_objs)
        else:
            assert "warning" not in idata.warmup_sample_stats
            assert "warning" not in idata.sample_stats
            assert "warning_dim_0" not in idata.warmup_sample_stats
            assert "warning_dim_0" not in idata.sample_stats


def test_sample_find_MAP_does_not_modify_start():
    # see https://github.com/pymc-devs/pymc/pull/4458
    with pm.Model():
        pm.LogNormal("untransformed")

        # make sure find_Map does not modify the start dict
        start = {"untransformed": 2}
        pm.find_MAP(start=start)
        assert start == {"untransformed": 2}

        # make sure sample does not modify the start dict
        start = {"untransformed": 0.2}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
            pm.sample(draws=10, step=pm.Metropolis(), tune=5, initvals=start, chains=3)
        assert start == {"untransformed": 0.2}

        # make sure sample does not modify the start when passes as list of dict
        start = [{"untransformed": 2}, {"untransformed": 0.2}]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
            pm.sample(draws=10, step=pm.Metropolis(), tune=5, initvals=start, chains=2)
        assert start == [{"untransformed": 2}, {"untransformed": 0.2}]


def test_empty_model():
    with pm.Model():
        pm.Normal("a", observed=1)
        with pytest.raises(SamplingError) as error:
            pm.sample()
        error.match("any free variables")


def test_partial_trace_unsupported():
    with pm.Model() as model:
        a = pm.Normal("a", mu=0, sigma=1)
        b = pm.Normal("b", mu=0, sigma=1)
        with pytest.raises(DeprecationWarning, match="removed support"):
            pm.sample(trace=[a])


@pytest.mark.xfail(condition=(pytensor.config.floatX == "float32"), reason="Fails on float32")
class TestNamedSampling:
    def test_shared_named(self):
        G_var = shared(value=np.atleast_2d(1.0), shape=(1, None), name="G")

        with pm.Model():
            theta0 = pm.Normal(
                "theta0",
                mu=np.atleast_2d(0),
                tau=np.atleast_2d(1e20),
                size=(1, 1),
                initval=np.atleast_2d(0),
            )
            theta = pm.Normal(
                "theta", mu=pt.dot(G_var, theta0), tau=np.atleast_2d(1e20), size=(1, 1)
            )
            res = theta.eval()
            assert np.isclose(res, 0.0)

    def test_shared_unnamed(self):
        G_var = shared(value=np.atleast_2d(1.0), shape=(1, None))
        with pm.Model():
            theta0 = pm.Normal(
                "theta0",
                mu=np.atleast_2d(0),
                tau=np.atleast_2d(1e20),
                size=(1, 1),
                initval=np.atleast_2d(0),
            )
            theta = pm.Normal(
                "theta", mu=pt.dot(G_var, theta0), tau=np.atleast_2d(1e20), size=(1, 1)
            )
            res = theta.eval()
            assert np.isclose(res, 0.0)

    def test_constant_named(self):
        G_var = pt.constant(np.atleast_2d(1.0), name="G")
        with pm.Model():
            theta0 = pm.Normal(
                "theta0",
                mu=np.atleast_2d(0),
                tau=np.atleast_2d(1e20),
                size=(1, 1),
                initval=np.atleast_2d(0),
            )
            theta = pm.Normal(
                "theta", mu=pt.dot(G_var, theta0), tau=np.atleast_2d(1e20), size=(1, 1)
            )

            res = theta.eval()
            assert np.isclose(res, 0.0)


def check_exec_nuts_init(method):
    with pm.Model() as model:
        pm.Normal("a", mu=0, sigma=1, size=2)
        pm.HalfNormal("b", sigma=1)
    with model:
        start, _ = pm.init_nuts(init=method, n_init=10, random_seed=[1])
        assert isinstance(start, list)
        assert len(start) == 1
        assert isinstance(start[0], dict)
        assert set(start[0].keys()) == {v.name for v in model.value_vars}
        start, _ = pm.init_nuts(init=method, n_init=10, chains=2, random_seed=[1, 2])
        assert isinstance(start, list)
        assert len(start) == 2
        assert isinstance(start[0], dict)
        assert set(start[0].keys()) == {v.name for v in model.value_vars}


@pytest.mark.parametrize(
    "method",
    [
        "advi",
        "ADVI+adapt_diag",
        "advi_map",
        "jitter+adapt_diag",
        "adapt_diag",
        "map",
        "adapt_full",
        "jitter+adapt_full",
    ],
)
def test_exec_nuts_init(method):
    if method.endswith("adapt_full"):
        with pytest.warns(UserWarning, match="experimental feature"):
            check_exec_nuts_init(method)
    else:
        check_exec_nuts_init(method)


@pytest.mark.skip(reason="Test requires monkey patching of RandomGenerator")
@pytest.mark.parametrize(
    "initval, jitter_max_retries, expectation",
    [
        (0, 0, pytest.raises(SamplingError)),
        (0, 1, pytest.raises(SamplingError)),
        (0, 4, does_not_raise()),
        (0, 10, does_not_raise()),
        (1, 0, does_not_raise()),
    ],
)
def test_init_jitter(initval, jitter_max_retries, expectation):
    with pm.Model() as m:
        pm.HalfNormal("x", transform=None, initval=initval)

    with expectation:
        # Starting value is negative (invalid) when np.random.rand returns 0 (jitter = -1)
        # and positive (valid) when it returns 1 (jitter = 1)
        with mock.patch("numpy.random.Generator.uniform", side_effect=[-1, -1, -1, 1, -1]):
            start = pm.sampling.mcmc._init_jitter(
                model=m,
                initvals=None,
                seeds=[1],
                jitter=True,
                jitter_max_retries=jitter_max_retries,
            )
            m.check_start_vals(start)


def test_step_args():
    with pm.Model() as model:
        a = pm.Normal("a")
        idata0 = pm.sample(target_accept=0.5, random_seed=1410)
        idata1 = pm.sample(nuts={"target_accept": 0.5}, random_seed=1410 * 2)
        idata2 = pm.sample(target_accept=0.5, nuts={"max_treedepth": 10}, random_seed=1410)

        with pytest.raises(ValueError, match="`target_accept` was defined twice."):
            pm.sample(target_accept=0.5, nuts={"target_accept": 0.95}, random_seed=1410)

    npt.assert_almost_equal(idata0.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)
    npt.assert_almost_equal(idata1.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)
    npt.assert_almost_equal(idata2.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)

    with pm.Model() as model:
        a = pm.Normal("a")
        b = pm.Poisson("b", 1)
        idata0 = pm.sample(target_accept=0.5, random_seed=1418)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "invalid value encountered in double_scalars", RuntimeWarning
            )
            idata1 = pm.sample(
                nuts={"target_accept": 0.5}, metropolis={"scaling": 0}, random_seed=1418 * 2
            )

    npt.assert_almost_equal(idata0.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)
    npt.assert_almost_equal(idata1.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)
    npt.assert_allclose(idata1.sample_stats.scaling, 0)


def test_init_nuts(caplog):
    with pm.Model() as model:
        a = pm.Normal("a")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
            pm.sample(10, tune=10)
        assert "Initializing NUTS" in caplog.text


def test_no_init_nuts_step(caplog):
    with pm.Model() as model:
        a = pm.Normal("a")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
            pm.sample(10, tune=10, step=pm.NUTS([a]))
        assert "Initializing NUTS" not in caplog.text


def test_no_init_nuts_compound(caplog):
    with pm.Model() as model:
        a = pm.Normal("a")
        b = pm.Poisson("b", 1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
            pm.sample(10, tune=10)
        assert "Initializing NUTS" not in caplog.text


def test_sample_var_names():
    with pm.Model() as model:
        a = pm.Normal("a")
        b = pm.Deterministic("b", a**2)
        idata = pm.sample(10, tune=10, var_names=["a"])
        assert "a" in idata.posterior
        assert "b" not in idata.posterior


class TestAssignStepMethods:
    def test_bernoulli(self):
        """Test bernoulli distribution is assigned binary gibbs metropolis method"""
        with pm.Model() as model:
            pm.Bernoulli("x", 0.5)
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, BinaryGibbsMetropolis)

    def test_normal(self):
        """Test normal distribution is assigned NUTS method"""
        with pm.Model() as model:
            pm.Normal("x", 0, 1)
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, NUTS)

    def test_categorical(self):
        """Test categorical distribution is assigned categorical gibbs metropolis method"""
        with pm.Model() as model:
            pm.Categorical("x", np.array([0.25, 0.75]))
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, BinaryGibbsMetropolis)
        with pm.Model() as model:
            pm.Categorical("y", np.array([0.25, 0.70, 0.05]))
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, CategoricalGibbsMetropolis)

    def test_binomial(self):
        """Test binomial distribution is assigned metropolis method."""
        with pm.Model() as model:
            pm.Binomial("x", 10, 0.5)
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, Metropolis)

    def test_normal_nograd_op(self):
        """Test normal distribution without an implemented gradient is assigned slice method"""
        with pm.Model() as model:
            x = pm.Normal("x", 0, 1)

            # a custom PyTensor Op that does not have a grad:
            is_64 = pytensor.config.floatX == "float64"
            itypes = [pt.dscalar] if is_64 else [pt.fscalar]
            otypes = [pt.dscalar] if is_64 else [pt.fscalar]

            @as_op(itypes, otypes)
            def kill_grad(x):
                return x

            data = np.random.normal(size=(100,))
            pm.Normal("y", mu=kill_grad(x), sigma=1, observed=data.astype(pytensor.config.floatX))

            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, Slice)

    @pytest.fixture
    def step_methods(self):
        """Make sure we reset the STEP_METHODS after the test is done."""
        methods_copy = pm.STEP_METHODS.copy()
        yield pm.STEP_METHODS
        pm.STEP_METHODS.clear()
        for method in methods_copy:
            pm.STEP_METHODS.append(method)

    def test_modify_step_methods(self, step_methods):
        """Test step methods can be changed"""
        step_methods.remove(NUTS)

        with pm.Model() as model:
            pm.Normal("x", 0, 1)
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert not isinstance(steps, NUTS)

        # add back nuts
        step_methods.append(NUTS)

        with pm.Model() as model:
            pm.Normal("x", 0, 1)
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, NUTS)

    def test_step_vars_in_model(self):
        """Test if error is raised if step variable is not found in model.value_vars"""
        with pm.Model() as model:
            c1 = pm.HalfNormal("c1")
            c2 = pm.HalfNormal("c2")

            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                step1 = NUTS([c1])
                step2 = NUTS([c2])
                step2.vars = [c2]
                step = CompoundStep([step1, step2])
                with pytest.raises(
                    ValueError,
                    match=r".* assigned to .* sampler is not a value variable in the model. You can use `util.get_value_vars_from_user_vars` to parse user provided variables.",
                ):
                    assign_step_methods(model, step)


class TestType:
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS)

    @pytensor.config.change_flags({"floatX": "float64", "warn_float64": "ignore"})
    def test_float64(self):
        with pm.Model() as model:
            x = pm.Normal("x", initval=np.array(1.0, dtype="float64"))
            obs = pm.Normal("obs", mu=x, sigma=1.0, observed=np.random.randn(5))

        assert x.dtype == "float64"
        assert obs.dtype == "float64"

        for sampler in self.samplers:
            with model:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                    pm.sample(draws=10, tune=10, chains=1, step=sampler())

    @pytensor.config.change_flags({"floatX": "float32", "warn_float64": "warn"})
    def test_float32(self):
        with pm.Model() as model:
            x = pm.Normal("x", initval=np.array(1.0, dtype="float32"))
            obs = pm.Normal("obs", mu=x, sigma=1.0, observed=np.random.randn(5).astype("float32"))

        assert x.dtype == "float32"
        assert obs.dtype == "float32"

        for sampler in self.samplers:
            with model:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                    pm.sample(draws=10, tune=10, chains=1, step=sampler())


class TestShared:
    def test_sample(self, seeded_test):
        x = np.random.normal(size=100)
        y = x + np.random.normal(scale=1e-2, size=100)

        x_pred = np.linspace(-3, 3, 200)

        x_shared = pytensor.shared(x)

        with pm.Model() as model:
            b = pm.Normal("b", 0.0, 10.0)
            pm.Normal("obs", b * x_shared, np.sqrt(1e-2), observed=y, shape=x_shared.shape)
            prior_trace0 = pm.sample_prior_predictive(1000)

            idata = pm.sample(1000, tune=1000, chains=1)
            pp_trace0 = pm.sample_posterior_predictive(idata)

            x_shared.set_value(x_pred)
            prior_trace1 = pm.sample_prior_predictive(1000)
            pp_trace1 = pm.sample_posterior_predictive(idata)

        assert prior_trace0.prior["b"].shape == (1, 1000)
        assert prior_trace0.prior_predictive["obs"].shape == (1, 1000, 100)
        np.testing.assert_allclose(
            x, pp_trace0.posterior_predictive["obs"].mean(("chain", "draw")), atol=1e-1
        )

        assert prior_trace1.prior["b"].shape == (1, 1000)
        assert prior_trace1.prior_predictive["obs"].shape == (1, 1000, 200)
        np.testing.assert_allclose(
            x_pred, pp_trace1.posterior_predictive["obs"].mean(("chain", "draw")), atol=1e-1
        )
