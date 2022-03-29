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

import unittest.mock as mock

from contextlib import ExitStack as does_not_raise
from itertools import combinations
from typing import Tuple

import aesara
import aesara.tensor as at
import numpy as np
import numpy.testing as npt
import pytest
import scipy.special

from aesara import shared
from arviz import InferenceData
from arviz import from_dict as az_from_dict
from scipy import stats

import pymc as pm

from pymc.aesaraf import compile_pymc
from pymc.backends.base import MultiTrace
from pymc.backends.ndarray import NDArray
from pymc.exceptions import IncorrectArgumentsError, SamplingError
from pymc.tests.helpers import SeededTest, fast_unstable_sampling_mode
from pymc.tests.models import simple_init


class TestInitNuts(SeededTest):
    def setup_method(self):
        super().setup_method()
        self.model, self.start, self.step, _ = simple_init()

    def test_checks_seeds_kwarg(self):
        with self.model:
            with pytest.raises(ValueError, match="must be array-like"):
                pm.sampling.init_nuts(seeds=1)
            with pytest.raises(ValueError, match="Number of seeds"):
                pm.sampling.init_nuts(chains=2, seeds=[1])


class TestSample(SeededTest):
    def setup_method(self):
        super().setup_method()
        self.model, self.start, self.step, _ = simple_init()

    def test_sample_does_not_set_seed(self):
        random_numbers = []
        for _ in range(2):
            np.random.seed(1)
            with self.model:
                pm.sample(1, tune=0, chains=1)
                random_numbers.append(np.random.random())
        assert random_numbers[0] == random_numbers[1]

    def test_parallel_sample_does_not_reuse_seed(self):
        cores = 4
        random_numbers = []
        draws = []
        for _ in range(2):
            np.random.seed(1)  # seeds in other processes don't effect main process
            with self.model:
                idata = pm.sample(100, tune=0, cores=cores)
            # numpy thread mentioned race condition.  might as well check none are equal
            for first, second in combinations(range(cores), 2):
                first_chain = idata.posterior["x"].sel(chain=first).values
                second_chain = idata.posterior["x"].sel(chain=second).values
                assert not np.allclose(first_chain, second_chain)
            draws.append(idata.posterior["x"].values)
            random_numbers.append(np.random.random())

        # Make sure future random processes aren't effected by this
        assert random_numbers[0] == random_numbers[1]
        assert (draws[0] == draws[1]).all()

    def test_sample(self):
        test_cores = [1]
        with self.model:
            for cores in test_cores:
                for steps in [1, 10, 300]:
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
                pm.sample(
                    init=init,
                    tune=120,
                    n_init=1000,
                    draws=50,
                    random_seed=self.random_seed,
                )

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
            idata = pm.sample(
                0,
                tune=5,
                cores=2,
                discard_tuned_samples=False,
                start=[{"x": [10, 10]}, {"x": [-10, -10]}],
                random_seed=self.random_seed,
            )
        assert idata.warmup_posterior["x"].sel(chain=0, draw=0).values[0] > 0
        assert idata.warmup_posterior["x"].sel(chain=1, draw=0).values[0] < 0

    def test_sample_tune_len(self):
        with self.model:
            trace = pm.sample(draws=100, tune=50, cores=1, return_inferencedata=False)
            assert len(trace) == 100
            trace = pm.sample(
                draws=100, tune=50, cores=1, return_inferencedata=False, discard_tuned_samples=False
            )
            assert len(trace) == 150
            trace = pm.sample(draws=100, tune=50, cores=4, return_inferencedata=False)
            assert len(trace) == 100

    def test_reset_tuning(self):
        with self.model:
            tune = 50
            chains = 2
            start, step = pm.sampling.init_nuts(chains=chains, seeds=[1, 2])
            pm.sample(draws=2, tune=tune, chains=chains, step=step, start=start, cores=1)
            assert step.potential._n_samples == tune
            assert step.step_adapt._count == tune + 1

    @pytest.mark.parametrize("step_cls", [pm.NUTS, pm.Metropolis, pm.Slice])
    @pytest.mark.parametrize("discard", [True, False])
    def test_trace_report(self, step_cls, discard):
        with self.model:
            # add more variables, because stats are 2D with CompoundStep!
            pm.Uniform("uni")
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
            pm.sample(10, cores=1, chains=2, trace=backend)

    def test_exceptions(self):
        # Test iteration over MultiTrace NotImplementedError
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            trace = pm.sample(tune=0, draws=10, chains=2, return_inferencedata=False)
            with pytest.raises(NotImplementedError):
                xvars = [t["mu"] for t in trace]

    def test_deterministic_of_unobserved(self):
        with pm.Model() as model:
            x = pm.HalfNormal("x", 1)
            y = pm.Deterministic("y", x + 100)
            idata = pm.sample(
                chains=1,
                tune=10,
                draws=50,
                compute_convergence_checks=False,
            )

        np.testing.assert_allclose(idata.posterior["y"], idata.posterior["x"] + 100)

    def test_transform_with_rv_dependency(self):
        # Test that untransformed variables that depend on upstream variables are properly handled
        with pm.Model() as m:
            x = pm.HalfNormal("x", observed=1)
            transform = pm.distributions.transforms.Interval(
                bounds_fn=lambda *inputs: (inputs[-2], inputs[-1])
            )
            y = pm.Uniform("y", lower=0, upper=x, transform=transform)
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
        pm.sample(draws=10, step=pm.Metropolis(), tune=5, start=start, chains=3)
        assert start == {"untransformed": 0.2}

        # make sure sample does not modify the start when passes as list of dict
        start = [{"untransformed": 2}, {"untransformed": 0.2}]
        pm.sample(draws=10, step=pm.Metropolis(), tune=5, start=start, chains=2)
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


def test_chain_idx():
    # see https://github.com/pymc-devs/pymc/issues/4469
    with pm.Model():
        mu = pm.Normal("mu")
        x = pm.Normal("x", mu=mu, sigma=1, observed=np.asarray(3))
        # note draws-tune must be >100 AND we need an observed RV for this to properly
        # trigger convergence checks, which is one particular case in which this failed
        # before
        idata = pm.sample(draws=150, tune=10, chain_idx=1)

        ppc = pm.sample_posterior_predictive(idata)
        # TODO FIXME: Assert something.
        ppc = pm.sample_posterior_predictive(idata, keep_size=True)


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
        G_var = shared(value=np.atleast_2d(1.0), broadcastable=(True, False), name="G")

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
        G_var = shared(value=np.atleast_2d(1.0), broadcastable=(True, False))
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
                return_inferencedata=False,
            )

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive(
                [model.compute_initial_point()], samples=10, return_inferencedata=False
            )
            # # deprecated argument is not introduced to fast version [2019/08/20:rpg]
            ppc = pm.sample_posterior_predictive(trace, var_names=["a"], return_inferencedata=False)
            # test empty ppc
            ppc = pm.sample_posterior_predictive(trace, var_names=[], return_inferencedata=False)
            assert len(ppc) == 0

            # test keep_size parameter
            ppc = pm.sample_posterior_predictive(trace, keep_size=True, return_inferencedata=False)
            assert ppc["a"].shape == (nchains, ndraws)

            # test default case
            ppc = pm.sample_posterior_predictive(trace, var_names=["a"], return_inferencedata=False)
            assert "a" in ppc
            assert ppc["a"].shape == (nchains * ndraws,)
            # mu's standard deviation may have changed thanks to a's observed
            _, pval = stats.kstest(ppc["a"] - trace["mu"], stats.norm(loc=0, scale=1).cdf)
            assert pval > 0.001

        # size argument not introduced to fast version [2019/08/20:rpg]
        with model:
            ppc = pm.sample_posterior_predictive(
                trace, size=5, var_names=["a"], return_inferencedata=False
            )
            assert ppc["a"].shape == (nchains * ndraws, 5)

    def test_normal_scalar_idata(self):
        nchains = 2
        ndraws = 500
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=0.0)
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

            ppc = pm.sample_posterior_predictive(idata, keep_size=True, return_inferencedata=False)
            assert ppc["a"].shape == (nchains, ndraws)

    def test_normal_vector(self, caplog):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            trace = pm.sample(return_inferencedata=False)

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive(
                [model.compute_initial_point()], return_inferencedata=False, samples=10
            )
            ppc = pm.sample_posterior_predictive(
                trace, return_inferencedata=False, samples=12, var_names=[]
            )
            assert len(ppc) == 0

            # test keep_size parameter
            ppc = pm.sample_posterior_predictive(trace, return_inferencedata=False, keep_size=True)
            assert ppc["a"].shape == (trace.nchains, len(trace), 2)
            with pytest.warns(UserWarning):
                ppc = pm.sample_posterior_predictive(
                    trace, return_inferencedata=False, samples=12, var_names=["a"]
                )
            assert "a" in ppc
            assert ppc["a"].shape == (12, 2)

            with pytest.warns(UserWarning):
                ppc = pm.sample_posterior_predictive(
                    trace, return_inferencedata=False, samples=12, var_names=["a"]
                )
            assert "a" in ppc
            assert ppc["a"].shape == (12, 2)

            # size unsupported by fast_ version  argument. [2019/08/19:rpg]
            ppc = pm.sample_posterior_predictive(
                trace, return_inferencedata=False, samples=10, var_names=["a"], size=4
            )
            assert "a" in ppc
            assert ppc["a"].shape == (10, 4, 2)

    def test_normal_vector_idata(self, caplog):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            trace = pm.sample(return_inferencedata=False)

        assert not isinstance(trace, InferenceData)

        with model:
            # test keep_size parameter with inference data as input...
            idata = pm.to_inference_data(trace)
            assert isinstance(idata, InferenceData)

            ppc = pm.sample_posterior_predictive(idata, return_inferencedata=False, keep_size=True)
            assert ppc["a"].shape == (trace.nchains, len(trace), 2)

    def test_exceptions(self, caplog):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            idata = pm.sample(idata_kwargs={"log_likelihood": False})

        with model:
            with pytest.raises(IncorrectArgumentsError):
                ppc = pm.sample_posterior_predictive(idata, samples=10, keep_size=True)

            with pytest.raises(IncorrectArgumentsError):
                ppc = pm.sample_posterior_predictive(idata, size=4, keep_size=True)

            # test wrong type argument
            bad_trace = {"mu": stats.norm.rvs(size=1000)}
            with pytest.raises(TypeError, match="type for `trace`"):
                ppc = pm.sample_posterior_predictive(bad_trace)

    def test_vector_observed(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=1)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.0, 1.0]))
            idata = pm.sample(idata_kwargs={"log_likelihood": False})

        with model:
            # test list input
            # ppc0 = pm.sample_posterior_predictive([model.initial_point], samples=10)
            # TODO: Assert something about the output
            # ppc = pm.sample_posterior_predictive(idata, samples=12, var_names=[])
            # assert len(ppc) == 0
            ppc = pm.sample_posterior_predictive(
                idata, return_inferencedata=False, samples=12, var_names=["a"]
            )
            assert "a" in ppc
            assert ppc["a"].shape == (12, 2)

            ppc = pm.sample_posterior_predictive(
                idata, return_inferencedata=False, samples=10, var_names=["a"], size=4
            )
            assert "a" in ppc
            assert ppc["a"].shape == (10, 4, 2)

    def test_sum_normal(self):
        with pm.Model() as model:
            a = pm.Normal("a", sigma=0.2)
            b = pm.Normal("b", mu=a)
            idata = pm.sample()

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive(
                [model.compute_initial_point()], return_inferencedata=False, samples=10
            )
            assert ppc0 == {}
            ppc = pm.sample_posterior_predictive(
                idata, return_inferencedata=False, samples=1000, var_names=["b"]
            )
            assert len(ppc) == 1
            assert ppc["b"].shape == (1000,)
            scale = np.sqrt(1 + 0.2**2)
            _, pval = stats.kstest(ppc["b"], stats.norm(scale=scale).cdf)
            assert pval > 0.001

    def test_model_not_drawable_prior(self):
        data = np.random.poisson(lam=10, size=200)
        model = pm.Model()
        with model:
            mu = pm.HalfFlat("sigma")
            pm.Poisson("foo", mu=mu, observed=data)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                idata = pm.sample(tune=10, draws=40, chains=1)

        with model:
            with pytest.raises(NotImplementedError) as excinfo:
                pm.sample_prior_predictive(50)
            assert "Cannot sample" in str(excinfo.value)
            samples = pm.sample_posterior_predictive(idata, 40, return_inferencedata=False)
            assert samples["foo"].shape == (40, 200)

    def test_model_shared_variable(self):
        rng = np.random.RandomState(9832)

        x = rng.randn(100)
        y = x > 0
        x_shared = aesara.shared(x)
        y_shared = aesara.shared(y)
        with pm.Model(rng_seeder=rng) as model:
            coeff = pm.Normal("x", mu=0, sigma=1)
            logistic = pm.Deterministic("p", pm.math.sigmoid(coeff * x_shared))

            obs = pm.Bernoulli("obs", p=logistic, observed=y_shared)
            trace = pm.sample(100, return_inferencedata=False, compute_convergence_checks=False)

        x_shared.set_value([-1, 0, 1.0])
        y_shared.set_value([0, 0, 0])

        samples = 100
        with model:
            post_pred = pm.sample_posterior_predictive(
                trace, return_inferencedata=False, samples=samples, var_names=["p", "obs"]
            )

        expected_p = np.array([logistic.eval({coeff: val}) for val in trace["x"][:samples]])
        assert post_pred["obs"].shape == (samples, 3)
        npt.assert_allclose(post_pred["p"], expected_p)

    def test_deterministic_of_observed(self):
        rng = np.random.RandomState(8442)

        meas_in_1 = pm.aesaraf.floatX(2 + 4 * rng.randn(10))
        meas_in_2 = pm.aesaraf.floatX(5 + 4 * rng.randn(10))
        nchains = 2
        with pm.Model(rng_seeder=rng) as model:
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
                )

            rtol = 1e-5 if aesara.config.floatX == "float64" else 1e-4

            ppc = pm.sample_posterior_predictive(
                return_inferencedata=False,
                model=model,
                trace=trace,
                samples=len(trace) * nchains,
                random_seed=0,
                var_names=[var.name for var in (model.deterministics + model.basic_RVs)],
            )

            npt.assert_allclose(ppc["in_1"] + ppc["in_2"], ppc["out"], rtol=rtol)

    def test_deterministic_of_observed_modified_interface(self):
        rng = np.random.RandomState(4982)

        meas_in_1 = pm.aesaraf.floatX(2 + 4 * rng.randn(100))
        meas_in_2 = pm.aesaraf.floatX(5 + 4 * rng.randn(100))
        with pm.Model(rng_seeder=rng) as model:
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
                )
            varnames = [v for v in trace.varnames if v != "out"]
            ppc_trace = [
                dict(zip(varnames, row)) for row in zip(*(trace.get_values(v) for v in varnames))
            ]
            ppc = pm.sample_posterior_predictive(
                return_inferencedata=False,
                model=model,
                trace=ppc_trace,
                samples=len(ppc_trace),
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
            ppc = pm.sample_posterior_predictive(trace, return_inferencedata=False, samples=1)
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


class TestSamplePPCW(SeededTest):
    @pytest.mark.xfail(reason="sample_posterior_predictive_w not refactored for v4")
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

    @pytest.mark.xfail(reason="sample_posterior_predictive_w not refactored for v4")
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
        start, _ = pm.init_nuts(init=method, n_init=10, seeds=[1])
        assert isinstance(start, list)
        assert len(start) == 1
        assert isinstance(start[0], dict)
        assert model.a.tag.value_var.name in start[0]
        assert model.b.tag.value_var.name in start[0]
        start, _ = pm.init_nuts(init=method, n_init=10, chains=2, seeds=[1, 2])
        assert isinstance(start, list)
        assert len(start) == 2
        assert isinstance(start[0], dict)
        assert model.a.tag.value_var.name in start[0]
        assert model.b.tag.value_var.name in start[0]


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
            burned_trace, return_inferencedata=False, samples=20, model=dm_model
        )
        assert sim_priors["probs"].shape == (20, 6)
        assert sim_priors["obs"].shape == (20,) + mn_data.shape
        assert sim_ppc["obs"].shape == (20,) + mn_data.shape

    def test_layers(self):
        with pm.Model(rng_seeder=232093) as model:
            a = pm.Uniform("a", lower=0, upper=1, size=10)
            b = pm.Binomial("b", n=1, p=a, size=10)

        b_sampler = compile_pymc([], b, mode="FAST_RUN")
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

        with pm.Model(rng_seeder=123) as model:
            ub = pm.HalfNormal("ub", 10)
            x = pm.Uniform("x", 0, ub)

            prior = pm.sample_prior_predictive(
                var_names=["ub", "ub_log__", "x", "x_interval__"],
                samples=10,
            )

        # Check values are correct
        assert np.allclose(prior.prior["ub_log__"].data, np.log(prior.prior["ub"].data))
        assert np.allclose(
            prior.prior["x_interval__"].data,
            ub_interval_forward(prior.prior["x"].data, prior.prior["ub"].data),
        )

        # Check that it works when the original RVs are not mentioned in var_names
        with pm.Model(rng_seeder=123) as model_transformed_only:
            ub = pm.HalfNormal("ub", 10)
            x = pm.Uniform("x", 0, ub)

            prior_transformed_only = pm.sample_prior_predictive(
                var_names=["ub_log__", "x_interval__"],
                samples=10,
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
        with pm.Model(rng_seeder=seed) as m1:
            a = pm.Normal("a")
            b = pm.Normal("b")
            c = pm.Normal("c")
            d = pm.Normal("d")
            prior1 = pm.sample_prior_predictive(samples=1, var_names=["a", "b", "c", "d"])

        with pm.Model(rng_seeder=seed) as m2:
            a = pm.Normal("a")
            b = pm.Normal("b")
            c = pm.Normal("c")
            d = pm.Normal("d")
            prior2 = pm.sample_prior_predictive(samples=1, var_names=["b", "a", "d", "c"])

        assert prior1.prior["a"] == prior2.prior["a"]
        assert prior1.prior["b"] == prior2.prior["b"]
        assert prior1.prior["c"] == prior2.prior["c"]
        assert prior1.prior["d"] == prior2.prior["d"]


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


class test_step_args(SeededTest):
    with pm.Model() as model:
        a = pm.Normal("a")
        idata0 = pm.sample(target_accept=0.5)
        idata1 = pm.sample(nuts={"target_accept": 0.5})

    npt.assert_almost_equal(idata0.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)
    npt.assert_almost_equal(idata1.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)

    with pm.Model() as model:
        a = pm.Normal("a")
        b = pm.Poisson("b", 1)
        idata0 = pm.sample(target_accept=0.5)
        idata1 = pm.sample(nuts={"target_accept": 0.5}, metropolis={"scaling": 0})

    npt.assert_almost_equal(idata0.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)
    npt.assert_almost_equal(idata1.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)
    npt.assert_allclose(idata1.sample_stats.scaling, 0)


def test_init_nuts(caplog):
    with pm.Model() as model:
        a = pm.Normal("a")
        pm.sample(10, tune=10)
        assert "Initializing NUTS" in caplog.text


def test_no_init_nuts_step(caplog):
    with pm.Model() as model:
        a = pm.Normal("a")
        pm.sample(10, tune=10, step=pm.NUTS([a]))
        assert "Initializing NUTS" not in caplog.text


def test_no_init_nuts_compound(caplog):
    with pm.Model() as model:
        a = pm.Normal("a")
        b = pm.Poisson("b", 1)
        pm.sample(10, tune=10)
        assert "Initializing NUTS" not in caplog.text
