#   Copyright 2020 The PyMC Developers
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
import re
import unittest.mock as mock
import warnings

from contextlib import ExitStack as does_not_raise
from copy import copy
from typing import Tuple

import aesara
import aesara.tensor as at
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import pytest
import scipy.special
import xarray as xr

from aesara import Mode, shared
from aesara.compile import SharedVariable
from aesara.compile.ops import as_op
from arviz import InferenceData
from arviz import from_dict as az_from_dict
from arviz.tests.helpers import check_multiple_attrs
from scipy import stats

import pymc as pm

from pymc.aesaraf import compile_pymc
from pymc.backends.base import MultiTrace
from pymc.backends.ndarray import NDArray
from pymc.distributions import transforms
from pymc.exceptions import SamplingError
from pymc.sampling import (
    _get_seeds_per_chain,
    assign_step_methods,
    compile_forward_sampling_function,
    get_vars_in_point_list,
)
from pymc.step_methods import (
    NUTS,
    BinaryGibbsMetropolis,
    CategoricalGibbsMetropolis,
    HamiltonianMC,
    Metropolis,
    Slice,
)
from pymc.tests.helpers import SeededTest, fast_unstable_sampling_mode
from pymc.tests.models import simple_init


class TestInitNuts(SeededTest):
    def setup_method(self):
        super().setup_method()
        self.model, self.start, self.step, _ = simple_init()

    def test_checks_seeds_kwarg(self):
        with self.model:
            with pytest.raises(ValueError, match="Number of seeds"):
                pm.sampling.init_nuts(chains=2, random_seed=[1])


class TestSample(SeededTest):
    def setup_method(self):
        super().setup_method()
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

    def test_sample(self):
        test_cores = [1]
        with self.model:
            for cores in test_cores:
                for steps in [1, 10, 300]:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                        warnings.filterwarnings(
                            "ignore", "More chains .* than draws .*", UserWarning
                        )
                        pm.sample(
                            steps,
                            tune=0,
                            step=self.step,
                            cores=cores,
                            random_seed=self.random_seed,
                        )

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
                    "random_seed": self.random_seed,
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
            with pytest.raises(ValueError) as excinfo:
                pm.sample(50, tune=0, foo=1)
            assert "'foo'" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                pm.sample(50, tune=0, foo={})
            assert "foo" in str(excinfo.value)

    def test_iter_sample(self):
        with self.model:
            samps = pm.sampling.iter_sample(
                draws=5,
                step=self.step,
                start=self.start,
                tune=0,
                random_seed=self.random_seed,
            )
            for i, trace in enumerate(samps):
                assert i == len(trace) - 1, "Trace does not have correct length."

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
                    random_seed=self.random_seed,
                )
        assert idata.warmup_posterior["x"].sel(chain=0, draw=0).values[0] > 0
        assert idata.warmup_posterior["x"].sel(chain=1, draw=0).values[0] < 0

    def test_sample_tune_len(self):
        with self.model:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                warnings.filterwarnings("ignore", "Tuning samples will be included.*", UserWarning)
                trace = pm.sample(draws=100, tune=50, cores=1, return_inferencedata=False)
                assert len(trace) == 100
                trace = pm.sample(
                    draws=100,
                    tune=50,
                    cores=1,
                    return_inferencedata=False,
                    discard_tuned_samples=False,
                )
                assert len(trace) == 150
                trace = pm.sample(draws=100, tune=50, cores=4, return_inferencedata=False)
                assert len(trace) == 100

    def test_reset_tuning(self):
        with self.model:
            tune = 50
            chains = 2
            start, step = pm.sampling.init_nuts(chains=chains, random_seed=[1, 2])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                pm.sample(draws=2, tune=tune, chains=chains, step=step, initvals=start, cores=1)
            assert step.potential._n_samples == tune
            assert step.step_adapt._count == tune + 1

    @pytest.mark.parametrize("step_cls", [pm.NUTS, pm.Metropolis, pm.Slice])
    @pytest.mark.parametrize("discard", [True, False])
    def test_trace_report(self, step_cls, discard):
        with self.model:
            # add more variables, because stats are 2D with CompoundStep!
            pm.Uniform("uni")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", ".*Tuning samples will be included.*", UserWarning
                )
                trace = pm.sample(
                    draws=100,
                    tune=50,
                    cores=1,
                    discard_tuned_samples=discard,
                    step=step_cls(),
                    compute_convergence_checks=False,
                    return_inferencedata=False,
                )
            assert trace.report.n_tune == 50
            assert trace.report.n_draws == 100
            assert isinstance(trace.report.t_sampling, float)

    def test_return_inferencedata(self):
        with self.model:
            kwargs = dict(draws=100, tune=50, cores=1, chains=2, step=pm.Metropolis())

            # trace with tuning
            with pytest.warns(UserWarning, match="will be included"):
                result = pm.sample(
                    **kwargs, return_inferencedata=False, discard_tuned_samples=False
                )
            assert isinstance(result, pm.backends.base.MultiTrace)
            assert len(result) == 150

            # inferencedata with tuning
            result = pm.sample(**kwargs, return_inferencedata=True, discard_tuned_samples=False)
            assert isinstance(result, InferenceData)
            assert result.posterior.sizes["draw"] == 100
            assert result.posterior.sizes["chain"] == 2
            assert len(result._groups_warmup) > 0

            # inferencedata without tuning, with idata_kwargs
            prior = pm.sample_prior_predictive(return_inferencedata=False)
            result = pm.sample(
                **kwargs,
                return_inferencedata=True,
                discard_tuned_samples=True,
                idata_kwargs={"prior": prior},
                random_seed=-1,
            )
            assert "prior" in result
            assert isinstance(result, InferenceData)
            assert result.posterior.sizes["draw"] == 100
            assert result.posterior.sizes["chain"] == 2
            assert len(result._groups_warmup) == 0

    @pytest.mark.parametrize("cores", [1, 2])
    def test_sampler_stat_tune(self, cores):
        with self.model:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                warnings.filterwarnings("ignore", "Tuning samples will be included.*", UserWarning)
                tune_stat = pm.sample(
                    tune=5,
                    draws=7,
                    cores=cores,
                    discard_tuned_samples=False,
                    return_inferencedata=False,
                    step=pm.Metropolis(),
                ).get_sampler_stats("tune", chains=1)
            assert list(tune_stat).count(True) == 5
            assert list(tune_stat).count(False) == 7

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
            pm.sampling._check_start_shape(self.model, start)

    @pytest.mark.parametrize("start", [{"x": np.array([1, 1])}, {"x": [10, 10]}, {"x": [-10, -10]}])
    def test_sample_start_good_shape(self, start):
        pm.sampling._check_start_shape(self.model, start)

    def test_sample_callback(self):
        callback = mock.Mock()
        test_cores = [1, 2]
        test_chains = [1, 2]
        with self.model:
            for cores in test_cores:
                for chain in test_chains:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                        pm.sample(
                            10,
                            tune=0,
                            chains=chain,
                            step=self.step,
                            cores=cores,
                            random_seed=self.random_seed,
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
                    random_seed=self.random_seed,
                    callback=callback,
                    return_inferencedata=False,
                )
            assert len(trace) == trace_cancel_length

    def test_sequential_backend(self):
        with self.model:
            backend = NDArray()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                pm.sample(10, cores=1, chains=2, trace=backend)

    def test_exceptions(self):
        # Test iteration over MultiTrace NotImplementedError
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                trace = pm.sample(tune=0, draws=10, chains=2, return_inferencedata=False)
            with pytest.raises(NotImplementedError):
                xvars = [t["mu"] for t in trace]

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


def test_partial_trace_sample():
    with pm.Model() as model:
        a = pm.Normal("a", mu=0, sigma=1)
        b = pm.Normal("b", mu=0, sigma=1)
        idata = pm.sample(trace=[a])
        assert "a" in idata.posterior
        assert "b" not in idata.posterior


@pytest.mark.parametrize(
    "n_points, tune, expected_length, expected_n_traces",
    [
        ((5, 2, 2), 0, 2, 3),
        ((6, 1, 1), 1, 6, 1),
    ],
)
def test_choose_chains(n_points, tune, expected_length, expected_n_traces):
    with pm.Model() as model:
        a = pm.Normal("a", mu=0, sigma=1)
        trace_0 = NDArray(model)
        trace_1 = NDArray(model)
        trace_2 = NDArray(model)
        trace_0.setup(n_points[0], 1)
        trace_1.setup(n_points[1], 1)
        trace_2.setup(n_points[2], 1)
        for _ in range(n_points[0]):
            trace_0.record({"a": 0})
        for _ in range(n_points[1]):
            trace_1.record({"a": 0})
        for _ in range(n_points[2]):
            trace_2.record({"a": 0})
        traces, length = pm.sampling._choose_chains([trace_0, trace_1, trace_2], tune=tune)
    assert length == expected_length
    assert expected_n_traces == len(traces)


@pytest.mark.xfail(condition=(aesara.config.floatX == "float32"), reason="Fails on float32")
class TestNamedSampling(SeededTest):
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
                "theta", mu=at.dot(G_var, theta0), tau=np.atleast_2d(1e20), size=(1, 1)
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
                "theta", mu=at.dot(G_var, theta0), tau=np.atleast_2d(1e20), size=(1, 1)
            )
            res = theta.eval()
            assert np.isclose(res, 0.0)

    def test_constant_named(self):
        G_var = at.constant(np.atleast_2d(1.0), name="G")
        with pm.Model():
            theta0 = pm.Normal(
                "theta0",
                mu=np.atleast_2d(0),
                tau=np.atleast_2d(1e20),
                size=(1, 1),
                initval=np.atleast_2d(0),
            )
            theta = pm.Normal(
                "theta", mu=at.dot(G_var, theta0), tau=np.atleast_2d(1e20), size=(1, 1)
            )

            res = theta.eval()
            assert np.isclose(res, 0.0)


class TestChooseBackend:
    def test_choose_backend_none(self):
        with mock.patch("pymc.sampling.NDArray") as nd:
            pm.sampling._choose_backend(None)
        assert nd.called

    def test_choose_backend_list_of_variables(self):
        with mock.patch("pymc.sampling.NDArray") as nd:
            pm.sampling._choose_backend(["var1", "var2"])
        nd.assert_called_with(vars=["var1", "var2"])

    def test_errors_and_warnings(self):
        with pm.Model():
            A = pm.Normal("A")
            B = pm.Uniform("B")
            strace = pm.sampling.NDArray(vars=[A, B])
            strace.setup(10, 0)

            with pytest.raises(ValueError, match="from existing MultiTrace"):
                pm.sampling._choose_backend(trace=MultiTrace([strace]))

            strace.record({"A": 2, "B_interval__": 0.1})
            assert len(strace) == 1
            with pytest.raises(ValueError, match="Continuation of traces"):
                pm.sampling._choose_backend(trace=strace)


class TestSamplePPC(SeededTest):
    def test_normal_scalar(self):
        nchains = 2
        ndraws = 500
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=0.0)
            trace = pm.sample(
                draws=ndraws,
                chains=nchains,
            )

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive(
                10 * [model.initial_point()], return_inferencedata=False
            )
            # # deprecated argument is not introduced to fast version [2019/08/20:rpg]
            ppc = pm.sample_posterior_predictive(trace, var_names=["a"], return_inferencedata=False)
            # test empty ppc
            ppc = pm.sample_posterior_predictive(trace, var_names=[], return_inferencedata=False)
            assert len(ppc) == 0

            # test keep_size parameter
            ppc = pm.sample_posterior_predictive(trace, return_inferencedata=False)
            assert ppc["a"].shape == (nchains, ndraws)

            # test default case
            idata_ppc = pm.sample_posterior_predictive(trace, var_names=["a"])
            ppc = idata_ppc.posterior_predictive
            assert "a" in ppc
            assert ppc["a"].shape == (nchains, ndraws)
            # mu's standard deviation may have changed thanks to a's observed
            _, pval = stats.kstest(
                (ppc["a"] - trace.posterior["mu"]).values.flatten(), stats.norm(loc=0, scale=1).cdf
            )
            assert pval > 0.001

    def test_normal_scalar_idata(self):
        nchains = 2
        ndraws = 500
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=0.0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Tuning samples will be included.*", UserWarning)
                trace = pm.sample(
                    draws=ndraws,
                    chains=nchains,
                    return_inferencedata=False,
                    discard_tuned_samples=False,
                )

        assert not isinstance(trace, InferenceData)

        with model:
            # test keep_size parameter and idata input
            idata = pm.to_inference_data(trace)
            assert isinstance(idata, InferenceData)

            ppc = pm.sample_posterior_predictive(idata, return_inferencedata=False)
            assert ppc["a"].shape == (nchains, ndraws)

    def test_normal_vector(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            trace = pm.sample(return_inferencedata=False, draws=12, chains=1)

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive(
                10 * [model.initial_point()],
                return_inferencedata=False,
            )
            ppc = pm.sample_posterior_predictive(trace, return_inferencedata=False, var_names=[])
            assert len(ppc) == 0

            ppc = pm.sample_posterior_predictive(trace, return_inferencedata=False)
            assert ppc["a"].shape == (trace.nchains, len(trace), 2)
            assert ppc0["a"].shape == (1, 10, 2)

    def test_normal_vector_idata(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            trace = pm.sample(return_inferencedata=False)

        assert not isinstance(trace, InferenceData)

        with model:
            # test keep_size parameter with inference data as input...
            idata = pm.to_inference_data(trace)
            assert isinstance(idata, InferenceData)

            ppc = pm.sample_posterior_predictive(idata, return_inferencedata=False)
            assert ppc["a"].shape == (trace.nchains, len(trace), 2)

    def test_exceptions(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            idata = pm.sample(idata_kwargs={"log_likelihood": False})

        with model:
            # test wrong type argument
            bad_trace = {"mu": stats.norm.rvs(size=1000)}
            with pytest.raises(TypeError, match="type for `trace`"):
                ppc = pm.sample_posterior_predictive(bad_trace)

    def test_sum_normal(self):
        with pm.Model() as model:
            a = pm.Normal("a", sigma=0.2)
            b = pm.Normal("b", mu=a)
            idata = pm.sample(draws=1000, chains=1)

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive(
                10 * [model.initial_point()], return_inferencedata=False
            )
            assert ppc0 == {}
            ppc = pm.sample_posterior_predictive(idata, return_inferencedata=False, var_names=["b"])
            assert len(ppc) == 1
            assert ppc["b"].shape == (
                1,
                1000,
            )
            scale = np.sqrt(1 + 0.2**2)
            _, pval = stats.kstest(ppc["b"].flatten(), stats.norm(scale=scale).cdf)
            assert pval > 0.001

    def test_model_not_drawable_prior(self):
        data = np.random.poisson(lam=10, size=200)
        model = pm.Model()
        with model:
            mu = pm.HalfFlat("sigma")
            pm.Poisson("foo", mu=mu, observed=data)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                    idata = pm.sample(tune=10, draws=40, chains=1)

        with model:
            with pytest.raises(NotImplementedError) as excinfo:
                pm.sample_prior_predictive(50)
            assert "Cannot sample" in str(excinfo.value)
            samples = pm.sample_posterior_predictive(idata, return_inferencedata=False)
            assert samples["foo"].shape == (1, 40, 200)

    def test_model_shared_variable(self):
        rng = np.random.RandomState(9832)

        x = rng.randn(100)
        y = x > 0
        x_shared = aesara.shared(x)
        y_shared = aesara.shared(y)
        samples = 100
        with pm.Model() as model:
            coeff = pm.Normal("x", mu=0, sigma=1)
            logistic = pm.Deterministic("p", pm.math.sigmoid(coeff * x_shared))

            obs = pm.Bernoulli("obs", p=logistic, observed=y_shared)
            trace = pm.sample(
                samples,
                chains=1,
                return_inferencedata=False,
                compute_convergence_checks=False,
                random_seed=rng,
            )

        x_shared.set_value([-1, 0, 1.0])
        y_shared.set_value([0, 0, 0])

        with model:
            post_pred = pm.sample_posterior_predictive(
                trace, return_inferencedata=False, var_names=["p", "obs"]
            )

        expected_p = np.array([[logistic.eval({coeff: val}) for val in trace["x"][:samples]]])
        assert post_pred["obs"].shape == (1, samples, 3)
        npt.assert_allclose(post_pred["p"], expected_p)

    def test_deterministic_of_observed(self):
        rng = np.random.RandomState(8442)

        meas_in_1 = pm.aesaraf.floatX(2 + 4 * rng.randn(10))
        meas_in_2 = pm.aesaraf.floatX(5 + 4 * rng.randn(10))
        nchains = 2
        with pm.Model() as model:
            mu_in_1 = pm.Normal("mu_in_1", 0, 2)
            sigma_in_1 = pm.HalfNormal("sd_in_1", 1)
            mu_in_2 = pm.Normal("mu_in_2", 0, 2)
            sigma_in_2 = pm.HalfNormal("sd__in_2", 1)

            in_1 = pm.Normal("in_1", mu_in_1, sigma_in_1, observed=meas_in_1)
            in_2 = pm.Normal("in_2", mu_in_2, sigma_in_2, observed=meas_in_2)
            out_diff = in_1 + in_2
            pm.Deterministic("out", out_diff)

            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                trace = pm.sample(
                    tune=100,
                    draws=100,
                    chains=nchains,
                    step=pm.Metropolis(),
                    return_inferencedata=False,
                    compute_convergence_checks=False,
                    random_seed=rng,
                )

            rtol = 1e-5 if aesara.config.floatX == "float64" else 1e-4

            ppc = pm.sample_posterior_predictive(
                return_inferencedata=False,
                model=model,
                trace=trace,
                random_seed=0,
                var_names=[var.name for var in (model.deterministics + model.basic_RVs)],
            )

            npt.assert_allclose(ppc["in_1"] + ppc["in_2"], ppc["out"], rtol=rtol)

    def test_deterministic_of_observed_modified_interface(self):
        rng = np.random.RandomState(4982)

        meas_in_1 = pm.aesaraf.floatX(2 + 4 * rng.randn(100))
        meas_in_2 = pm.aesaraf.floatX(5 + 4 * rng.randn(100))
        with pm.Model() as model:
            mu_in_1 = pm.Normal("mu_in_1", 0, 1, initval=0)
            sigma_in_1 = pm.HalfNormal("sd_in_1", 1, initval=1)
            mu_in_2 = pm.Normal("mu_in_2", 0, 1, initval=0)
            sigma_in_2 = pm.HalfNormal("sd__in_2", 1, initval=1)

            in_1 = pm.Normal("in_1", mu_in_1, sigma_in_1, observed=meas_in_1)
            in_2 = pm.Normal("in_2", mu_in_2, sigma_in_2, observed=meas_in_2)
            out_diff = in_1 + in_2
            pm.Deterministic("out", out_diff)

            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                trace = pm.sample(
                    tune=100,
                    draws=100,
                    step=pm.Metropolis(),
                    return_inferencedata=False,
                    compute_convergence_checks=False,
                    random_seed=rng,
                )
            varnames = [v for v in trace.varnames if v != "out"]
            ppc_trace = [
                dict(zip(varnames, row)) for row in zip(*(trace.get_values(v) for v in varnames))
            ]
            ppc = pm.sample_posterior_predictive(
                return_inferencedata=False,
                model=model,
                trace=ppc_trace,
                var_names=[x.name for x in (model.deterministics + model.basic_RVs)],
            )

            rtol = 1e-5 if aesara.config.floatX == "float64" else 1e-3
            npt.assert_allclose(ppc["in_1"] + ppc["in_2"], ppc["out"], rtol=rtol)

    def test_variable_type(self):
        with pm.Model() as model:
            mu = pm.HalfNormal("mu", 1)
            a = pm.Normal("a", mu=mu, sigma=2, observed=np.array([1, 2]))
            b = pm.Poisson("b", mu, observed=np.array([1, 2]))
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                trace = pm.sample(
                    tune=10, draws=10, compute_convergence_checks=False, return_inferencedata=False
                )

        with model:
            ppc = pm.sample_posterior_predictive(trace, return_inferencedata=False)
            assert ppc["a"].dtype.kind == "f"
            assert ppc["b"].dtype.kind == "i"

    def test_potentials_warning(self):
        warning_msg = "The effect of Potentials on other parameters is ignored during"
        with pm.Model() as m:
            a = pm.Normal("a", 0, 1)
            p = pm.Potential("p", a + 1)
            obs = pm.Normal("obs", a, 1, observed=5)

        trace = az_from_dict({"a": np.random.rand(5)})
        with m:
            with pytest.warns(UserWarning, match=warning_msg):
                pm.sample_posterior_predictive(trace)

    def test_idata_extension(self):
        """Testing if sample_posterior_predictive() extends inferenceData"""

        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=[0.0, 1.0])
            idata = pm.sample(tune=10, draws=10, compute_convergence_checks=False)

        base_test_dict = {
            "posterior": ["mu", "~a"],
            "sample_stats": ["diverging", "lp"],
            "log_likelihood": ["a"],
            "observed_data": ["a"],
        }
        test_dict = {"~posterior_predictive": [], "~predictions": [], **base_test_dict}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

        # extending idata with in-sample ppc
        with model:
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        # test addition
        test_dict = {"posterior_predictive": ["a"], "~predictions": [], **base_test_dict}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

        # extending idata with out-of-sample ppc
        with model:
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, predictions=True)
        # test addition
        test_dict = {"posterior_predictive": ["a"], "predictions": ["a"], **base_test_dict}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

    @pytest.mark.parametrize("multitrace", [False, True])
    def test_deterministics_out_of_idata(self, multitrace):
        draws = 10
        chains = 2
        coords = {"draw": range(draws), "chain": range(chains)}
        ds = xr.Dataset(
            {
                "a": xr.DataArray(
                    [[0] * draws] * chains,
                    coords=coords,
                    dims=["chain", "draw"],
                )
            },
            coords=coords,
        )
        with pm.Model() as m:
            a = pm.Normal("a")
            if multitrace:
                straces = []
                for chain in ds.chain:
                    strace = pm.backends.NDArray(model=m, vars=[a])
                    strace.setup(len(ds.draw), int(chain))
                    strace.values = {"a": ds.a.sel(chain=chain).data}
                    strace.draw_idx = len(ds.draw)
                    straces.append(strace)
                trace = MultiTrace(straces)
            else:
                trace = ds

            d = pm.Deterministic("d", a - 4)
            pm.Normal("c", d, sigma=0.01)
            ppc = pm.sample_posterior_predictive(trace, var_names="c", return_inferencedata=True)
        assert np.all(np.abs(ppc.posterior_predictive.c + 4) <= 0.1)

    def test_logging_sampled_basic_rvs_prior(self, caplog):
        with pm.Model() as m:
            x = pm.Normal("x")
            y = pm.Deterministic("y", x + 1)
            z = pm.Normal("z", y, observed=0)

        with m:
            pm.sample_prior_predictive(samples=1)
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [x, z]")]
        caplog.clear()

        with m:
            pm.sample_prior_predictive(samples=1, var_names=["x"])
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [x]")]
        caplog.clear()

    def test_logging_sampled_basic_rvs_posterior(self, caplog):
        with pm.Model() as m:
            x = pm.Normal("x")
            x_det = pm.Deterministic("x_det", x + 1)
            y = pm.Normal("y", x_det)
            z = pm.Normal("z", y, observed=0)

        idata = az_from_dict(posterior={"x": np.zeros(5), "x_det": np.ones(5), "y": np.ones(5)})
        with m:
            pm.sample_posterior_predictive(idata)
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [z]")]
        caplog.clear()

        with m:
            pm.sample_posterior_predictive(idata, var_names=["y", "z"])
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [y, z]")]
        caplog.clear()

        # Resampling `x` will force resampling of `y`, even if it is in trace
        with m:
            pm.sample_posterior_predictive(idata, var_names=["x", "z"])
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [x, y, z]")]
        caplog.clear()

        # Missing deterministic `x_det` does not show in the log, even if it is being
        # recomputed, only `y` RV shows
        idata = az_from_dict(posterior={"x": np.zeros(5)})
        with m:
            pm.sample_posterior_predictive(idata)
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [y, z]")]
        caplog.clear()

        # Missing deterministic `x_det` does not cause recomputation of downstream `y` RV
        idata = az_from_dict(posterior={"x": np.zeros(5), "y": np.ones(5)})
        with m:
            pm.sample_posterior_predictive(idata)
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [z]")]
        caplog.clear()

        # Missing `x` causes sampling of downstream `y` RV, even if it is present in trace
        idata = az_from_dict(posterior={"y": np.ones(5)})
        with m:
            pm.sample_posterior_predictive(idata)
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [x, y, z]")]
        caplog.clear()

    def test_logging_sampled_basic_rvs_posterior_deterministic(self, caplog):
        with pm.Model() as m:
            x = pm.Normal("x")
            x_det = pm.Deterministic("x_det", x + 1)
            y = pm.Normal("y", x_det)
            z = pm.Normal("z", y, observed=0)

        # Explicit resampling a deterministic will lead to resampling of downstream RV `y`
        # This behavior could change in the future as the posterior of `y` is still valid
        idata = az_from_dict(posterior={"x": np.zeros(5), "x_det": np.ones(5), "y": np.ones(5)})
        with m:
            pm.sample_posterior_predictive(idata, var_names=["x_det", "z"])
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [y, z]")]
        caplog.clear()

    @staticmethod
    def make_mock_model():
        rng = np.random.default_rng(seed=42)
        data = rng.normal(loc=1, scale=0.2, size=(10, 3))
        with pm.Model() as model:
            model.add_coord("name", ["A", "B", "C"], mutable=True)
            model.add_coord("obs", list(range(10, 20)), mutable=True)
            offsets = pm.MutableData("offsets", rng.normal(0, 1, size=(10,)))
            a = pm.Normal("a", mu=0, sigma=1, dims=["name"])
            b = pm.Normal("b", mu=offsets, sigma=1)
            mu = pm.Deterministic("mu", a + b[..., None], dims=["obs", "name"])
            sigma = pm.HalfNormal("sigma", sigma=1, dims=["name"])

            data = pm.MutableData(
                "y_obs",
                data,
                dims=["obs", "name"],
            )
            pm.Normal("y", mu=mu, sigma=sigma, observed=data, dims=["obs", "name"])
        return model

    @pytest.fixture(scope="class")
    def mock_multitrace(self):
        with self.make_mock_model():
            trace = pm.sample(
                draws=10,
                tune=10,
                chains=2,
                progressbar=False,
                compute_convergence_checks=False,
                return_inferencedata=False,
                random_seed=42,
            )
        return trace

    @pytest.fixture(scope="class", params=["MultiTrace", "InferenceData", "Dataset"])
    def mock_sample_results(self, request, mock_multitrace):
        kind = request.param
        trace = mock_multitrace
        # We rebuild the class to ensure that all dimensions, data and coords start out
        # the same across params values
        model = self.make_mock_model()
        if kind == "MultiTrace":
            return kind, trace, model
        else:
            idata = pm.to_inference_data(
                trace,
                save_warmup=False,
                model=model,
                log_likelihood=False,
            )
            if kind == "Dataset":
                return kind, idata.posterior, model
            else:
                return kind, idata, model

    def test_logging_sampled_basic_rvs_posterior_mutable(self, mock_sample_results, caplog):
        kind, samples, model = mock_sample_results
        with model:
            pm.sample_posterior_predictive(samples)
        if kind == "MultiTrace":
            # MultiTrace will only have the actual MCMC posterior samples but no information on
            # the MutableData and mutable coordinate values, so it will always assume they are volatile
            # and resample their descendants
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()
        elif kind == "InferenceData":
            # InferenceData has all MCMC posterior samples and the values for both coordinates and
            # data containers. This enables it to see that no data has changed and it should only
            # resample the observed variable
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [y]")]
            caplog.clear()
        elif kind == "Dataset":
            # Dataset has all MCMC posterior samples and the values of the coordinates. This
            # enables it to see that the coordinates have not changed, but the MutableData is
            # assumed volatile by default
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [b, y]")]
            caplog.clear()

        original_offsets = model["offsets"].get_value()
        with model:
            # Changing the MutableData values. This will only be picked up by InferenceData
            pm.set_data({"offsets": original_offsets + 1})
            pm.sample_posterior_predictive(samples)
        if kind == "MultiTrace":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()
        elif kind == "InferenceData":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [b, y]")]
            caplog.clear()
        elif kind == "Dataset":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [b, y]")]
            caplog.clear()

        with model:
            # Changing the mutable coordinates. This will be picked up by InferenceData and Dataset
            model.set_dim("name", new_length=4, coord_values=["D", "E", "F", "G"])
            pm.set_data({"offsets": original_offsets, "y_obs": np.zeros((10, 4))})
            pm.sample_posterior_predictive(samples)
        if kind == "MultiTrace":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()
        elif kind == "InferenceData":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, sigma, y]")]
            caplog.clear()
        elif kind == "Dataset":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()

        with model:
            # Changing the mutable coordinate values, but not shape, and also changing MutableData.
            # This will trigger resampling of all variables
            model.set_dim("name", new_length=3, coord_values=["A", "B", "D"])
            pm.set_data({"offsets": original_offsets + 1, "y_obs": np.zeros((10, 3))})
            pm.sample_posterior_predictive(samples)
        if kind == "MultiTrace":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()
        elif kind == "InferenceData":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()
        elif kind == "Dataset":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()


@pytest.mark.xfail(
    reason="sample_posterior_predictive_w not refactored for v4", raises=NotImplementedError
)
class TestSamplePPCW(SeededTest):
    def test_sample_posterior_predictive_w(self):
        data0 = np.random.normal(0, 1, size=50)
        warning_msg = "The number of samples is too small to check convergence reliably"

        with pm.Model() as model_0:
            mu = pm.Normal("mu", mu=0, sigma=1)
            y = pm.Normal("y", mu=mu, sigma=1, observed=data0)
            with pytest.warns(UserWarning, match=warning_msg):
                trace_0 = pm.sample(10, tune=0, chains=2, return_inferencedata=False)
            idata_0 = pm.to_inference_data(trace_0, log_likelihood=False)

        with pm.Model() as model_1:
            mu = pm.Normal("mu", mu=0, sigma=1, size=len(data0))
            y = pm.Normal("y", mu=mu, sigma=1, observed=data0)
            with pytest.warns(UserWarning, match=warning_msg):
                trace_1 = pm.sample(10, tune=0, chains=2, return_inferencedata=False)
            idata_1 = pm.to_inference_data(trace_1, log_likelihood=False)

        with pm.Model() as model_2:
            # Model with no observed RVs.
            mu = pm.Normal("mu", mu=0, sigma=1)
            with pytest.warns(UserWarning, match=warning_msg):
                trace_2 = pm.sample(10, tune=0, return_inferencedata=False)

        traces = [trace_0, trace_1]
        idatas = [idata_0, idata_1]
        models = [model_0, model_1]

        ppc = pm.sample_posterior_predictive_w(traces, 100, models)
        assert ppc["y"].shape == (100, 50)

        ppc = pm.sample_posterior_predictive_w(idatas, 100, models)
        assert ppc["y"].shape == (100, 50)

        with model_0:
            ppc = pm.sample_posterior_predictive_w([idata_0.posterior], None)
            assert ppc["y"].shape == (20, 50)

        with pytest.raises(ValueError, match="The number of traces and weights should be the same"):
            pm.sample_posterior_predictive_w([idata_0.posterior], 100, models, weights=[0.5, 0.5])

        with pytest.raises(ValueError, match="The number of models and weights should be the same"):
            pm.sample_posterior_predictive_w([idata_0.posterior], 100, models)

        with pytest.raises(
            ValueError, match="The number of observed RVs should be the same for all models"
        ):
            pm.sample_posterior_predictive_w([trace_0, trace_2], 100, [model_0, model_2])

    def test_potentials_warning(self):
        warning_msg = "The effect of Potentials on other parameters is ignored during"
        with pm.Model() as m:
            a = pm.Normal("a", 0, 1)
            p = pm.Potential("p", a + 1)
            obs = pm.Normal("obs", a, 1, observed=5)

        trace = az_from_dict({"a": np.random.rand(10)})
        with pytest.warns(UserWarning, match=warning_msg):
            pm.sample_posterior_predictive_w(samples=5, traces=[trace, trace], models=[m, m])


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
            start = pm.sampling._init_jitter(
                model=m,
                initvals=None,
                seeds=[1],
                jitter=True,
                jitter_max_retries=jitter_max_retries,
            )
            m.check_start_vals(start)


@pytest.fixture(scope="class")
def point_list_arg_bug_fixture() -> Tuple[pm.Model, pm.backends.base.MultiTrace]:
    with pm.Model() as pmodel:
        n = pm.Normal("n")
        trace = pm.sample(return_inferencedata=False)

    with pmodel:
        d = pm.Deterministic("d", n * 4)
    return pmodel, trace


class TestSamplePriorPredictive(SeededTest):
    def test_ignores_observed(self):
        observed = np.random.normal(10, 1, size=200)
        with pm.Model():
            # Use a prior that's way off to show we're ignoring the observed variables
            observed_data = pm.MutableData("observed_data", observed)
            mu = pm.Normal("mu", mu=-100, sigma=1)
            positive_mu = pm.Deterministic("positive_mu", np.abs(mu))
            z = -1 - positive_mu
            pm.Normal("x_obs", mu=z, sigma=1, observed=observed_data)
            prior = pm.sample_prior_predictive(return_inferencedata=False)

        assert "observed_data" not in prior
        assert (prior["mu"] < -90).all()
        assert (prior["positive_mu"] > 90).all()
        assert (prior["x_obs"] < -90).all()
        assert prior["x_obs"].shape == (500, 200)
        npt.assert_array_almost_equal(prior["positive_mu"], np.abs(prior["mu"]), decimal=4)

    def test_respects_shape(self):
        for shape in (2, (2,), (10, 2), (10, 10)):
            with pm.Model():
                mu = pm.Gamma("mu", 3, 1, size=1)
                goals = pm.Poisson("goals", mu, size=shape)
                trace1 = pm.sample_prior_predictive(
                    10, return_inferencedata=False, var_names=["mu", "mu", "goals"]
                )
                trace2 = pm.sample_prior_predictive(
                    10, return_inferencedata=False, var_names=["mu", "goals"]
                )
            if shape == 2:  # want to test shape as an int
                shape = (2,)
            assert trace1["goals"].shape == (10,) + shape
            assert trace2["goals"].shape == (10,) + shape

    def test_multivariate(self):
        with pm.Model():
            m = pm.Multinomial("m", n=5, p=np.array([0.25, 0.25, 0.25, 0.25]))
            trace = pm.sample_prior_predictive(10)

        assert trace.prior["m"].shape == (1, 10, 4)

    def test_multivariate2(self):
        # Added test for issue #3271
        mn_data = np.random.multinomial(n=100, pvals=[1 / 6.0] * 6, size=10)
        with pm.Model() as dm_model:
            probs = pm.Dirichlet("probs", a=np.ones(6))
            obs = pm.Multinomial("obs", n=100, p=probs, observed=mn_data)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                burned_trace = pm.sample(
                    tune=10,
                    draws=20,
                    chains=1,
                    return_inferencedata=False,
                    compute_convergence_checks=False,
                )
        sim_priors = pm.sample_prior_predictive(
            return_inferencedata=False, samples=20, model=dm_model
        )
        sim_ppc = pm.sample_posterior_predictive(
            burned_trace, return_inferencedata=False, model=dm_model
        )
        assert sim_priors["probs"].shape == (20, 6)
        assert sim_priors["obs"].shape == (20,) + mn_data.shape
        assert sim_ppc["obs"].shape == (1, 20) + mn_data.shape

    def test_layers(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0, upper=1, size=10)
            b = pm.Binomial("b", n=1, p=a, size=10)

        b_sampler = compile_pymc([], b, mode="FAST_RUN", random_seed=232093)
        avg = np.stack([b_sampler() for i in range(10000)]).mean(0)
        npt.assert_array_almost_equal(avg, 0.5 * np.ones((10,)), decimal=2)

    def test_transformed(self):
        n = 18
        at_bats = 45 * np.ones(n, dtype=int)
        hits = np.random.randint(1, 40, size=n, dtype=int)
        draws = 50

        with pm.Model() as model:
            phi = pm.Beta("phi", alpha=1.0, beta=1.0)

            kappa_log = pm.Exponential("logkappa", lam=5.0)
            kappa = pm.Deterministic("kappa", at.exp(kappa_log))

            thetas = pm.Beta("thetas", alpha=phi * kappa, beta=(1.0 - phi) * kappa, size=n)

            y = pm.Binomial("y", n=at_bats, p=thetas, observed=hits)
            gen = pm.sample_prior_predictive(draws)

        assert gen.prior["phi"].shape == (1, draws)
        assert gen.prior_predictive["y"].shape == (1, draws, n)
        assert "thetas" in gen.prior.data_vars

    def test_shared(self):
        n1 = 10
        obs = shared(np.random.rand(n1) < 0.5)
        draws = 50

        with pm.Model() as m:
            p = pm.Beta("p", 1.0, 1.0)
            y = pm.Bernoulli("y", p, observed=obs)
            o = pm.Deterministic("o", obs)
            gen1 = pm.sample_prior_predictive(draws)

        assert gen1.prior_predictive["y"].shape == (1, draws, n1)
        assert gen1.prior["o"].shape == (1, draws, n1)

        n2 = 20
        obs.set_value(np.random.rand(n2) < 0.5)
        with m:
            gen2 = pm.sample_prior_predictive(draws)

        assert gen2.prior_predictive["y"].shape == (1, draws, n2)
        assert gen2.prior["o"].shape == (1, draws, n2)

    def test_density_dist(self):
        obs = np.random.normal(-1, 0.1, size=10)
        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            sigma = pm.HalfNormal("sigma", 1e-6)
            a = pm.DensityDist(
                "a",
                mu,
                sigma,
                random=lambda mu, sigma, rng=None, size=None: rng.normal(
                    loc=mu, scale=sigma, size=size
                ),
                observed=obs,
            )
            prior = pm.sample_prior_predictive(return_inferencedata=False)

        npt.assert_almost_equal((prior["a"] - prior["mu"][..., None]).mean(), 0, decimal=3)

    def test_shape_edgecase(self):
        with pm.Model():
            mu = pm.Normal("mu", size=5)
            sigma = pm.Uniform("sigma", lower=2, upper=3)
            x = pm.Normal("x", mu=mu, sigma=sigma, size=5)
            prior = pm.sample_prior_predictive(10)
        assert prior.prior["mu"].shape == (1, 10, 5)

    def test_zeroinflatedpoisson(self):
        with pm.Model():
            mu = pm.Beta("mu", alpha=1, beta=1)
            psi = pm.HalfNormal("psi", sigma=1)
            pm.ZeroInflatedPoisson("suppliers", psi=psi, mu=mu, size=20)
            gen_data = pm.sample_prior_predictive(samples=5000)
            assert gen_data.prior["mu"].shape == (1, 5000)
            assert gen_data.prior["psi"].shape == (1, 5000)
            assert gen_data.prior["suppliers"].shape == (1, 5000, 20)

    def test_potentials_warning(self):
        warning_msg = "The effect of Potentials on other parameters is ignored during"
        with pm.Model() as m:
            a = pm.Normal("a", 0, 1)
            p = pm.Potential("p", a + 1)

        with m:
            with pytest.warns(UserWarning, match=warning_msg):
                pm.sample_prior_predictive(samples=5)

    def test_transformed_vars(self):
        # Test that prior predictive returns transformation of RVs when these are
        # passed explicitly in `var_names`

        def ub_interval_forward(x, ub):
            # Interval transform assuming lower bound is zero
            return np.log(x - 0) - np.log(ub - x)

        with pm.Model() as model:
            ub = pm.HalfNormal("ub", 10)
            x = pm.Uniform("x", 0, ub)

            prior = pm.sample_prior_predictive(
                var_names=["ub", "ub_log__", "x", "x_interval__"],
                samples=10,
                random_seed=123,
            )

        # Check values are correct
        assert np.allclose(prior.prior["ub_log__"].data, np.log(prior.prior["ub"].data))
        assert np.allclose(
            prior.prior["x_interval__"].data,
            ub_interval_forward(prior.prior["x"].data, prior.prior["ub"].data),
        )

        # Check that it works when the original RVs are not mentioned in var_names
        with pm.Model() as model_transformed_only:
            ub = pm.HalfNormal("ub", 10)
            x = pm.Uniform("x", 0, ub)

            prior_transformed_only = pm.sample_prior_predictive(
                var_names=["ub_log__", "x_interval__"],
                samples=10,
                random_seed=123,
            )
        assert (
            "ub" not in prior_transformed_only.prior.data_vars
            and "x" not in prior_transformed_only.prior.data_vars
        )
        assert np.allclose(
            prior.prior["ub_log__"].data, prior_transformed_only.prior["ub_log__"].data
        )
        assert np.allclose(
            prior.prior["x_interval__"], prior_transformed_only.prior["x_interval__"].data
        )

    def test_issue_4490(self):
        # Test that samples do not depend on var_name order or, more fundamentally,
        # that they do not depend on the set order used inside `sample_prior_predictive`
        seed = 4490
        with pm.Model() as m1:
            a = pm.Normal("a")
            b = pm.Normal("b")
            c = pm.Normal("c")
            d = pm.Normal("d")
            prior1 = pm.sample_prior_predictive(
                samples=1, var_names=["a", "b", "c", "d"], random_seed=seed
            )

        with pm.Model() as m2:
            a = pm.Normal("a")
            b = pm.Normal("b")
            c = pm.Normal("c")
            d = pm.Normal("d")
            prior2 = pm.sample_prior_predictive(
                samples=1, var_names=["b", "a", "d", "c"], random_seed=seed
            )

        assert prior1.prior["a"] == prior2.prior["a"]
        assert prior1.prior["b"] == prior2.prior["b"]
        assert prior1.prior["c"] == prior2.prior["c"]
        assert prior1.prior["d"] == prior2.prior["d"]

    def test_aesara_function_kwargs(self):
        sharedvar = aesara.shared(0)
        with pm.Model() as m:
            x = pm.DiracDelta("x", 0)
            y = pm.Deterministic("y", x + sharedvar)

            prior = pm.sample_prior_predictive(
                samples=5,
                return_inferencedata=False,
                compile_kwargs=dict(
                    mode=Mode("py"),
                    updates={sharedvar: sharedvar + 1},
                ),
            )

        assert np.all(prior["y"] == np.arange(5))


class TestSamplePosteriorPredictive:
    def test_point_list_arg_bug_spp(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture
        with pmodel:
            pp = pm.sample_posterior_predictive(
                [trace[15]], return_inferencedata=False, var_names=["d"]
            )

    def test_sample_from_xarray_prior(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture

        with pmodel:
            prior = pm.sample_prior_predictive(
                samples=20,
                return_inferencedata=False,
            )
            idat = pm.to_inference_data(trace, prior=prior)

        with pmodel:
            pp = pm.sample_posterior_predictive(
                idat.prior, return_inferencedata=False, var_names=["d"]
            )

    def test_sample_from_xarray_posterior(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture
        with pmodel:
            idat = pm.to_inference_data(trace)
            pp = pm.sample_posterior_predictive(idat.posterior, var_names=["d"])

    def test_aesara_function_kwargs(self):
        sharedvar = aesara.shared(0)
        with pm.Model() as m:
            x = pm.DiracDelta("x", 0.0)
            y = pm.Deterministic("y", x + sharedvar)

            pp = pm.sample_posterior_predictive(
                trace=az_from_dict({"x": np.arange(5)}),
                var_names=["y"],
                return_inferencedata=False,
                compile_kwargs=dict(
                    mode=Mode("py"),
                    updates={sharedvar: sharedvar + 1},
                ),
            )

        assert np.all(pp["y"] == np.arange(5) * 2)

    def test_sample_dims(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture
        with pmodel:
            post = pm.to_inference_data(trace).posterior.stack(sample=["chain", "draw"])
            pp = pm.sample_posterior_predictive(post, var_names=["d"], sample_dims=["sample"])
            assert "sample" in pp.posterior_predictive
            assert len(pp.posterior_predictive["sample"]) == len(post["sample"])
            post = post.expand_dims(pp_dim=5)
            pp = pm.sample_posterior_predictive(
                post, var_names=["d"], sample_dims=["sample", "pp_dim"]
            )
            assert "sample" in pp.posterior_predictive
            assert "pp_dim" in pp.posterior_predictive
            assert len(pp.posterior_predictive["sample"]) == len(post["sample"])
            assert len(pp.posterior_predictive["pp_dim"]) == 5


class TestDraw(SeededTest):
    def test_univariate(self):
        with pm.Model():
            x = pm.Normal("x")

        x_draws = pm.draw(x)
        assert x_draws.shape == ()

        (x_draws,) = pm.draw([x])
        assert x_draws.shape == ()

        x_draws = pm.draw(x, draws=10)
        assert x_draws.shape == (10,)

        (x_draws,) = pm.draw([x], draws=10)
        assert x_draws.shape == (10,)

    def test_multivariate(self):
        with pm.Model():
            mln = pm.Multinomial("mln", n=5, p=np.array([0.25, 0.25, 0.25, 0.25]))

        mln_draws = pm.draw(mln, draws=1)
        assert mln_draws.shape == (4,)

        (mln_draws,) = pm.draw([mln], draws=1)
        assert mln_draws.shape == (4,)

        mln_draws = pm.draw(mln, draws=10)
        assert mln_draws.shape == (10, 4)

        (mln_draws,) = pm.draw([mln], draws=10)
        assert mln_draws.shape == (10, 4)

    def test_multiple_variables(self):
        with pm.Model():
            x = pm.Normal("x")
            y = pm.Normal("y", shape=10)
            z = pm.Uniform("z", shape=5)
            w = pm.Dirichlet("w", a=[1, 1, 1])

        num_draws = 100
        draws = pm.draw((x, y, z, w), draws=num_draws)
        assert draws[0].shape == (num_draws,)
        assert draws[1].shape == (num_draws, 10)
        assert draws[2].shape == (num_draws, 5)
        assert draws[3].shape == (num_draws, 3)

    def test_draw_different_samples(self):
        with pm.Model():
            x = pm.Normal("x")

        x_draws_1 = pm.draw(x, 100)
        x_draws_2 = pm.draw(x, 100)
        assert not np.all(np.isclose(x_draws_1, x_draws_2))

    def test_draw_aesara_function_kwargs(self):
        sharedvar = aesara.shared(0)
        x = pm.DiracDelta.dist(0.0)
        y = x + sharedvar
        draws = pm.draw(
            y,
            draws=5,
            mode=Mode("py"),
            updates={sharedvar: sharedvar + 1},
        )
        assert np.all(draws == np.arange(5))


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


class TestCompileForwardSampler:
    @staticmethod
    def get_function_roots(function):
        return [
            var
            for var in aesara.graph.basic.graph_inputs(function.maker.fgraph.outputs)
            if var.name
        ]

    @staticmethod
    def get_function_inputs(function):
        return {i for i in function.maker.fgraph.inputs if not isinstance(i, SharedVariable)}

    def test_linear_model(self):
        with pm.Model() as model:
            x = pm.MutableData("x", np.linspace(0, 1, 10))
            y = pm.MutableData("y", np.ones(10))

            alpha = pm.Normal("alpha", 0, 0.1)
            beta = pm.Normal("beta", 0, 0.1)
            mu = pm.Deterministic("mu", alpha + beta * x)
            sigma = pm.HalfNormal("sigma", 0.1)
            obs = pm.Normal("obs", mu, sigma, observed=y, shape=x.shape)

        f, volatile_rvs = compile_forward_sampling_function(
            [obs],
            vars_in_trace=[alpha, beta, sigma, mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"alpha", "beta", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"x", "alpha", "beta", "sigma"}

        with pm.Model() as model:
            x = pm.ConstantData("x", np.linspace(0, 1, 10))
            y = pm.MutableData("y", np.ones(10))

            alpha = pm.Normal("alpha", 0, 0.1)
            beta = pm.Normal("beta", 0, 0.1)
            mu = pm.Deterministic("mu", alpha + beta * x)
            sigma = pm.HalfNormal("sigma", 0.1)
            obs = pm.Normal("obs", mu, sigma, observed=y, shape=x.shape)

        f, volatile_rvs = compile_forward_sampling_function(
            [obs],
            vars_in_trace=[alpha, beta, sigma, mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"alpha", "beta", "sigma", "mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu", "sigma"}

    def test_nested_observed_model(self):
        with pm.Model() as model:
            p = pm.ConstantData("p", np.array([0.25, 0.5, 0.25]))
            x = pm.MutableData("x", np.zeros(10))
            y = pm.MutableData("y", np.ones(10))

            category = pm.Categorical("category", p, observed=x)
            beta = pm.Normal("beta", 0, 0.1, size=p.shape)
            mu = pm.Deterministic("mu", beta[category])
            sigma = pm.HalfNormal("sigma", 0.1)
            obs = pm.Normal("obs", mu, sigma, observed=y, shape=mu.shape)

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[beta, mu, sigma],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {category, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"beta", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"x", "p", "beta", "sigma"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[beta, mu, sigma],
            basic_rvs=model.basic_RVs,
            givens_dict={category: np.zeros(10, dtype=category.dtype)},
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"beta", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {
            "x",
            "p",
            "category",
            "beta",
            "sigma",
        }

    def test_volatile_parameters(self):
        with pm.Model() as model:
            y = pm.MutableData("y", np.ones(10))
            mu = pm.Normal("mu", 0, 1)
            nested_mu = pm.Normal("nested_mu", mu, 1, size=10)
            sigma = pm.HalfNormal("sigma", 1)
            obs = pm.Normal("obs", nested_mu, sigma, observed=y, shape=nested_mu.shape)

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[nested_mu, sigma],  # mu isn't in the trace and will be deemed volatile
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {mu, nested_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"sigma"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[mu, nested_mu, sigma],
            basic_rvs=model.basic_RVs,
            givens_dict={
                mu: np.array(1.0)
            },  # mu will be considered volatile because it's in givens
        )
        assert volatile_rvs == {nested_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu", "sigma"}

    def test_mixture(self):
        with pm.Model() as model:
            w = pm.Dirichlet("w", a=np.ones(3), size=(5, 3))

            mu = pm.Normal("mu", mu=np.arange(3), sigma=1)

            components = pm.Normal.dist(mu=mu, sigma=1, size=w.shape)
            mix_mu = pm.Mixture("mix_mu", w=w, comp_dists=components)
            obs = pm.Normal("obs", mix_mu, 1, observed=np.ones((5, 3)))

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mix_mu, mu, w],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"w", "mu", "mix_mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"mix_mu"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mu, w],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {mix_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"w", "mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"w", "mu"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mix_mu, mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {w, mix_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu"}

    def test_censored(self):
        with pm.Model() as model:
            latent_mu = pm.Normal("latent_mu", mu=np.arange(3), sigma=1)
            mu = pm.Censored("mu", pm.Normal.dist(mu=latent_mu, sigma=1), lower=-1, upper=1)
            obs = pm.Normal("obs", mu, 1, observed=np.ones((10, 3)))

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[latent_mu, mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"latent_mu", "mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {latent_mu, mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == set()

    def test_lkj_cholesky_cov(self):
        with pm.Model() as model:
            mu = np.zeros(3)
            sd_dist = pm.Exponential.dist(1.0, size=3)
            chol, corr, stds = pm.LKJCholeskyCov(  # pylint: disable=unpacking-non-sequence
                "chol_packed", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            chol_packed = model["chol_packed"]
            chol = pm.Deterministic("chol", chol)
            obs = pm.MvNormal("obs", mu=mu, chol=chol, observed=np.zeros(3))

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[chol_packed, chol],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"chol_packed", "chol"}
        assert {i.name for i in self.get_function_roots(f)} == {"chol"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[chol_packed],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"chol_packed"}
        assert {i.name for i in self.get_function_roots(f)} == {"chol_packed"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[chol],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {chol_packed, obs}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == set()

    def test_non_random_model_variable(self):
        with pm.Model() as model:
            # A user may register non-pure RandomVariables that can nevertheless be
            # sampled, as long as a custom logprob is dispatched or Aeppl can infer
            # its logprob (which is the case for `clip`)
            y = at.clip(pm.Normal.dist(), -1, 1)
            y = model.register_rv(y, name="y")
            y_abs = pm.Deterministic("y_abs", at.abs(y))
            obs = pm.Normal("obs", y_abs, observed=np.zeros(10))

        # y_abs should be resampled even if in the trace, because the source y is missing
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[y_abs],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {y, obs}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == set()

    def test_mutable_coords_volatile(self):
        rng = np.random.default_rng(seed=42)
        data = rng.normal(loc=1, scale=0.2, size=(10, 3))
        with pm.Model() as model:
            model.add_coord("name", ["A", "B", "C"], mutable=True)
            model.add_coord("obs", list(range(10, 20)), mutable=True)
            offsets = pm.MutableData("offsets", rng.normal(0, 1, size=(10,)))
            a = pm.Normal("a", mu=0, sigma=1, dims=["name"])
            b = pm.Normal("b", mu=offsets, sigma=1)
            mu = pm.Deterministic("mu", a + b[..., None], dims=["obs", "name"])
            sigma = pm.HalfNormal("sigma", sigma=1, dims=["name"])

            data = pm.MutableData(
                "y_obs",
                data,
                dims=["obs", "name"],
            )
            y = pm.Normal("y", mu=mu, sigma=sigma, observed=data, dims=["obs", "name"])

        # When no constant_data and constant_coords, all the dependent nodes will be volatile and
        # resampled
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {y, a, b, sigma}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == {"name", "obs", "offsets"}

        # When the constant data has the same values as the shared data, offsets wont be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_data={"offsets": offsets.get_value()},
        )
        assert volatile_rvs == {y, a, sigma}
        assert {i.name for i in self.get_function_inputs(f)} == {"b"}
        assert {i.name for i in self.get_function_roots(f)} == {"b", "name", "obs"}

        # When we declare constant_coords, the shared variables with matching names wont be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_coords={"name", "obs"},
        )
        assert volatile_rvs == {y, b}
        assert {i.name for i in self.get_function_inputs(f)} == {"a", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {
            "a",
            "sigma",
            "name",
            "obs",
            "offsets",
        }

        # When we have both constant_data and constant_coords, only y will be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_data={"offsets": offsets.get_value()},
            constant_coords={"name", "obs"},
        )
        assert volatile_rvs == {y}
        assert {i.name for i in self.get_function_inputs(f)} == {"a", "b", "mu", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu", "sigma", "name", "obs"}

        # When constant_data has different values than the shared variable, then
        # offsets will be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_data={"offsets": offsets.get_value() + 1},
            constant_coords={"name", "obs"},
        )
        assert volatile_rvs == {y, b}
        assert {i.name for i in self.get_function_inputs(f)} == {"a", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {
            "a",
            "sigma",
            "name",
            "obs",
            "offsets",
        }


def test_get_seeds_per_chain():
    ret = _get_seeds_per_chain(None, chains=1)
    assert len(ret) == 1 and isinstance(ret[0], int)

    ret = _get_seeds_per_chain(None, chains=2)
    assert len(ret) == 2 and isinstance(ret[0], int)

    ret = _get_seeds_per_chain(5, chains=1)
    assert ret == (5,)

    ret = _get_seeds_per_chain(5, chains=3)
    assert len(ret) == 3 and isinstance(ret[0], int) and not any(r == 5 for r in ret)

    rng = np.random.default_rng(123)
    expected_ret = rng.integers(2**30, dtype=np.int64, size=1)
    rng = np.random.default_rng(123)
    ret = _get_seeds_per_chain(rng, chains=1)
    assert ret == expected_ret

    rng = np.random.RandomState(456)
    expected_ret = rng.randint(2**30, dtype=np.int64, size=2)
    rng = np.random.RandomState(456)
    ret = _get_seeds_per_chain(rng, chains=2)
    assert np.all(ret == expected_ret)

    for expected_ret in ([0, 1, 2], (0, 1, 2, 3), np.arange(5)):
        ret = _get_seeds_per_chain(expected_ret, chains=len(expected_ret))
        assert ret is expected_ret

        with pytest.raises(ValueError, match="does not match the number of chains"):
            _get_seeds_per_chain(expected_ret, chains=len(expected_ret) + 1)

    with pytest.raises(ValueError, match=re.escape("The `seeds` must be array-like")):
        _get_seeds_per_chain({1: 1, 2: 2}, 2)


def test_distinct_rvs():
    """Make sure `RandomVariable`s generated using a `Model`'s default RNG state all have distinct states."""

    with pm.Model() as model:
        X_rv = pm.Normal("x")
        Y_rv = pm.Normal("y")

        pp_samples = pm.sample_prior_predictive(
            samples=2, return_inferencedata=False, random_seed=npr.RandomState(2023532)
        )

    assert X_rv.owner.inputs[0] != Y_rv.owner.inputs[0]

    with pm.Model():
        X_rv = pm.Normal("x")
        Y_rv = pm.Normal("y")

        pp_samples_2 = pm.sample_prior_predictive(
            samples=2, return_inferencedata=False, random_seed=npr.RandomState(2023532)
        )

    assert np.array_equal(pp_samples["y"], pp_samples_2["y"])


class TestNestedRandom(SeededTest):
    def build_model(self, distribution, shape, nested_rvs_info):
        with pm.Model() as model:
            nested_rvs = {}
            for rv_name, info in nested_rvs_info.items():
                try:
                    value, nested_shape = info
                    loc = 0.0
                except ValueError:
                    value, nested_shape, loc = info
                if value is None:
                    nested_rvs[rv_name] = pm.Uniform(
                        rv_name,
                        0 + loc,
                        1 + loc,
                        shape=nested_shape,
                    )
                else:
                    nested_rvs[rv_name] = value * np.ones(nested_shape)
            rv = distribution(
                "target",
                shape=shape,
                **nested_rvs,
            )
        return model, rv, nested_rvs

    def sample_prior(self, distribution, shape, nested_rvs_info, prior_samples):
        model, rv, nested_rvs = self.build_model(
            distribution,
            shape,
            nested_rvs_info,
        )
        with model:
            return pm.sample_prior_predictive(prior_samples, return_inferencedata=False)

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "mu", "alpha"],
        [
            [10, (3,), (None, tuple()), (None, (3,))],
            [10, (3,), (None, (3,)), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_NegativeBinomial(
        self,
        prior_samples,
        shape,
        mu,
        alpha,
    ):
        prior = self.sample_prior(
            distribution=pm.NegativeBinomial,
            shape=shape,
            nested_rvs_info=dict(mu=mu, alpha=alpha),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "psi", "mu", "alpha"],
        [
            [10, (3,), (0.5, tuple()), (None, tuple()), (None, (3,))],
            [10, (3,), (0.5, (3,)), (None, tuple()), (None, (3,))],
            [10, (3,), (0.5, tuple()), (None, (3,)), (None, tuple())],
            [10, (3,), (0.5, (3,)), (None, (3,)), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.5, (3,)),
                (None, (3,)),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.5, (3,)),
                (None, (3,)),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_ZeroInflatedNegativeBinomial(
        self,
        prior_samples,
        shape,
        psi,
        mu,
        alpha,
    ):
        prior = self.sample_prior(
            distribution=pm.ZeroInflatedNegativeBinomial,
            shape=shape,
            nested_rvs_info=dict(psi=psi, mu=mu, alpha=alpha),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "nu", "sigma"],
        [
            [10, (3,), (None, tuple()), (None, (3,))],
            [10, (3,), (None, tuple()), (None, (3,))],
            [10, (3,), (None, (3,)), (None, tuple())],
            [10, (3,), (None, (3,)), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_Rice(
        self,
        prior_samples,
        shape,
        nu,
        sigma,
    ):
        prior = self.sample_prior(
            distribution=pm.Rice,
            shape=shape,
            nested_rvs_info=dict(nu=nu, sigma=sigma),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "mu", "sigma", "lower", "upper"],
        [
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, (3,), -1), (None, tuple())],
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, (3,), -1), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (1.0, tuple()),
                (None, (3,), -1),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (1.0, tuple()),
                (None, (3,), -1),
                (None, (4, 3)),
            ],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, (3,), -1), (None, tuple())],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, (3,), -1), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.0, tuple()),
                (None, (3,)),
                (None, (3,), -1),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.0, tuple()),
                (None, (3,)),
                (None, (3,), -1),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_TruncatedNormal(
        self,
        prior_samples,
        shape,
        mu,
        sigma,
        lower,
        upper,
    ):
        prior = self.sample_prior(
            distribution=pm.TruncatedNormal,
            shape=shape,
            nested_rvs_info=dict(mu=mu, sigma=sigma, lower=lower, upper=upper),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "c", "lower", "upper"],
        [
            [10, (3,), (None, tuple()), (-1.0, (3,)), (2, tuple())],
            [10, (3,), (None, tuple()), (-1.0, tuple()), (None, tuple(), 1)],
            [10, (3,), (None, (3,)), (-1.0, tuple()), (None, tuple(), 1)],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (-1.0, tuple()),
                (None, (3,), 1),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, tuple(), -1),
                (None, (3,), 1),
            ],
        ],
        ids=str,
    )
    def test_Triangular(
        self,
        prior_samples,
        shape,
        c,
        lower,
        upper,
    ):
        prior = self.sample_prior(
            distribution=pm.Triangular,
            shape=shape,
            nested_rvs_info=dict(c=c, lower=lower, upper=upper),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape


class TestAssignStepMethods:
    def test_bernoulli(self):
        """Test bernoulli distribution is assigned binary gibbs metropolis method"""
        with pm.Model() as model:
            pm.Bernoulli("x", 0.5)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, BinaryGibbsMetropolis)

    def test_normal(self):
        """Test normal distribution is assigned NUTS method"""
        with pm.Model() as model:
            pm.Normal("x", 0, 1)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, NUTS)

    def test_categorical(self):
        """Test categorical distribution is assigned categorical gibbs metropolis method"""
        with pm.Model() as model:
            pm.Categorical("x", np.array([0.25, 0.75]))
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, BinaryGibbsMetropolis)
        with pm.Model() as model:
            pm.Categorical("y", np.array([0.25, 0.70, 0.05]))
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, CategoricalGibbsMetropolis)

    def test_binomial(self):
        """Test binomial distribution is assigned metropolis method."""
        with pm.Model() as model:
            pm.Binomial("x", 10, 0.5)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, Metropolis)

    def test_normal_nograd_op(self):
        """Test normal distribution without an implemented gradient is assigned slice method"""
        with pm.Model() as model:
            x = pm.Normal("x", 0, 1)

            # a custom Aesara Op that does not have a grad:
            is_64 = aesara.config.floatX == "float64"
            itypes = [at.dscalar] if is_64 else [at.fscalar]
            otypes = [at.dscalar] if is_64 else [at.fscalar]

            @as_op(itypes, otypes)
            def kill_grad(x):
                return x

            data = np.random.normal(size=(100,))
            pm.Normal("y", mu=kill_grad(x), sigma=1, observed=data.astype(aesara.config.floatX))

            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, Slice)

    def test_modify_step_methods(self):
        """Test step methods can be changed"""
        # remove nuts from step_methods
        step_methods = list(pm.STEP_METHODS)
        step_methods.remove(NUTS)
        pm.STEP_METHODS = step_methods

        with pm.Model() as model:
            pm.Normal("x", 0, 1)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert not isinstance(steps, NUTS)

        # add back nuts
        pm.STEP_METHODS = step_methods + [NUTS]

        with pm.Model() as model:
            pm.Normal("x", 0, 1)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, NUTS)


class TestType:
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS)

    def setup_method(self):
        # save Aesara config object
        self.aesara_config = copy(aesara.config)

    def teardown_method(self):
        # restore aesara config
        aesara.config = self.aesara_config

    @aesara.config.change_flags({"floatX": "float64", "warn_float64": "ignore"})
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

    @aesara.config.change_flags({"floatX": "float32", "warn_float64": "warn"})
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


class TestShared(SeededTest):
    def test_sample(self):
        x = np.random.normal(size=100)
        y = x + np.random.normal(scale=1e-2, size=100)

        x_pred = np.linspace(-3, 3, 200)

        x_shared = aesara.shared(x)

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


def test_get_vars_in_point_list():
    with pm.Model() as modelA:
        pm.Normal("a", 0, 1)
        pm.Normal("b", 0, 1)
    with pm.Model() as modelB:
        a = pm.Normal("a", 0, 1)
        pm.Normal("c", 0, 1)

    point_list = [{"a": 0, "b": 0}]
    vars_in_trace = get_vars_in_point_list(point_list, modelB)
    assert set(vars_in_trace) == {a}

    strace = pm.backends.NDArray(model=modelB, vars=modelA.free_RVs)
    strace.setup(1, 1)
    strace.values = point_list[0]
    strace.draw_idx = 1
    trace = MultiTrace([strace])
    vars_in_trace = get_vars_in_point_list(trace, modelB)
    assert set(vars_in_trace) == {a}
