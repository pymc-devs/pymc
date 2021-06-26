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

import arviz as az
import numpy as np
import numpy.testing as npt
import pytest
import theano
import theano.tensor as tt

from scipy import stats
from theano import shared

import pymc3 as pm

from pymc3.backends.ndarray import NDArray
from pymc3.exceptions import IncorrectArgumentsError, SamplingError
from pymc3.tests.helpers import SeededTest
from pymc3.tests.models import simple_init


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestSample(SeededTest):
    def setup_method(self):
        super().setup_method()
        self.model, self.start, self.step, _ = simple_init()

    def test_sample_does_not_set_seed(self):
        random_numbers = []
        for _ in range(2):
            np.random.seed(1)
            with self.model:
                pm.sample(1, tune=0, chains=1, return_inferencedata=False)
                random_numbers.append(np.random.random())
        assert random_numbers[0] == random_numbers[1]

    def test_parallel_sample_does_not_reuse_seed(self):
        cores = 4
        random_numbers = []
        draws = []
        for _ in range(2):
            np.random.seed(1)  # seeds in other processes don't effect main process
            with self.model:
                trace = pm.sample(100, tune=0, cores=cores, return_inferencedata=False)
            # numpy thread mentioned race condition.  might as well check none are equal
            for first, second in combinations(range(cores), 2):
                first_chain = trace.get_values("x", chains=first)
                second_chain = trace.get_values("x", chains=second)
                assert not (first_chain == second_chain).all()
            draws.append(trace.get_values("x"))
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
                        return_inferencedata=False,
                    )

    def test_sample_init(self):
        with self.model:
            for init in ("advi", "advi_map", "map"):
                pm.sample(
                    init=init,
                    tune=0,
                    n_init=1000,
                    draws=50,
                    random_seed=self.random_seed,
                    return_inferencedata=False,
                )

    def test_sample_args(self):
        with self.model:
            with pytest.raises(ValueError) as excinfo:
                pm.sample(50, tune=0, init=None, foo=1, return_inferencedata=False)
            assert "'foo'" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                pm.sample(50, tune=0, init=None, foo={}, return_inferencedata=False)
            assert "foo" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                pm.sample(10, tune=0, init=None, target_accept=0.9, return_inferencedata=False)
            assert "target_accept" in str(excinfo.value)

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
            tr = pm.sample(
                0,
                tune=5,
                cores=2,
                discard_tuned_samples=False,
                start=[{"x": [10, 10]}, {"x": [-10, -10]}],
                random_seed=self.random_seed,
            )
        assert tr.get_values("x", chains=0)[0][0] > 0
        assert tr.get_values("x", chains=1)[0][0] < 0

    def test_sample_tune_len(self):
        with self.model:
            trace = pm.sample(draws=100, tune=50, cores=1, return_inferencedata=False)
            assert len(trace) == 100
            trace = pm.sample(
                draws=100, tune=50, cores=1, discard_tuned_samples=False, return_inferencedata=False
            )
            assert len(trace) == 150
            trace = pm.sample(draws=100, tune=50, cores=4, return_inferencedata=False)
            assert len(trace) == 100

    def test_reset_tuning(self):
        with self.model:
            tune = 50
            chains = 2
            start, step = pm.sampling.init_nuts(chains=chains)
            pm.sample(
                draws=2,
                tune=tune,
                chains=chains,
                step=step,
                start=start,
                cores=1,
                return_inferencedata=False,
            )
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
                return_inferencedata=False,
            )
            assert trace.report.n_tune == 50
            assert trace.report.n_draws == 100
            assert isinstance(trace.report.t_sampling, float)
        pass

    def test_trace_report_bart(self):
        X = np.random.normal(0, 1, size=(3, 250)).T
        Y = np.random.normal(0, 1, size=250)
        X[:, 0] = np.random.normal(Y, 0.1)

        with pm.Model() as model:
            mu = pm.BART("mu", X, Y, m=20)
            sigma = pm.HalfNormal("sigma", 1)
            y = pm.Normal("y", mu, sigma, observed=Y)
            trace = pm.sample(500, tune=100, random_seed=3415, return_inferencedata=False)
        var_imp = trace.report.variable_importance
        assert var_imp[0] > var_imp[1:].sum()
        npt.assert_almost_equal(var_imp.sum(), 1)

    def test_return_inferencedata(self, monkeypatch):
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
            assert isinstance(result, az.InferenceData)
            assert result.posterior.sizes["draw"] == 100
            assert result.posterior.sizes["chain"] == 2
            assert len(result._groups_warmup) > 0

            # inferencedata without tuning, with idata_kwargs
            prior = pm.sample_prior_predictive()
            result = pm.sample(
                **kwargs,
                return_inferencedata=True,
                discard_tuned_samples=True,
                idata_kwargs={"prior": prior},
                random_seed=-1
            )
            assert "prior" in result
            assert isinstance(result, az.InferenceData)
            assert result.posterior.sizes["draw"] == 100
            assert result.posterior.sizes["chain"] == 2
            assert len(result._groups_warmup) == 0

            # check warning for version 3.10 onwards
            monkeypatch.setattr("pymc3.__version__", "3.10")
            with pytest.warns(FutureWarning, match="pass return_inferencedata"):
                result = pm.sample(**kwargs)
        pass

    @pytest.mark.parametrize("cores", [1, 2])
    def test_sampler_stat_tune(self, cores):
        with self.model:
            tune_stat = pm.sample(
                tune=5,
                draws=7,
                cores=cores,
                discard_tuned_samples=False,
                step=pm.Metropolis(),
                return_inferencedata=False,
            ).get_sampler_stats("tune", chains=1)
            assert list(tune_stat).count(True) == 5
            assert list(tune_stat).count(False) == 7
        pass

    @pytest.mark.parametrize(
        "start, error",
        [
            ([1, 2], TypeError),
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
                        return_inferencedata=False,
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


def test_sample_find_MAP_does_not_modify_start():
    # see https://github.com/pymc-devs/pymc3/pull/4458
    with pm.Model():
        pm.Lognormal("untransformed")

        # make sure find_Map does not modify the start dict
        start = {"untransformed": 2}
        pm.find_MAP(start=start)
        assert start == {"untransformed": 2}

        # make sure sample does not modify the start dict
        start = {"untransformed": 0.2}
        pm.sample(
            draws=10,
            step=pm.Metropolis(),
            tune=5,
            start=start,
            chains=3,
            return_inferencedata=False,
        )
        assert start == {"untransformed": 0.2}

        # make sure sample does not modify the start when passes as list of dict
        start = [{"untransformed": 2}, {"untransformed": 0.2}]
        pm.sample(
            draws=10,
            step=pm.Metropolis(),
            tune=5,
            start=start,
            chains=2,
            return_inferencedata=False,
        )
        assert start == [{"untransformed": 2}, {"untransformed": 0.2}]


def test_empty_model():
    with pm.Model():
        pm.Normal("a", observed=1)
        with pytest.raises(ValueError) as error:
            pm.sample(return_inferencedata=False)
        error.match("any free variables")


def test_partial_trace_sample():
    with pm.Model() as model:
        a = pm.Normal("a", mu=0, sigma=1)
        b = pm.Normal("b", mu=0, sigma=1)
        trace = pm.sample(trace=[a], return_inferencedata=False)


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


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestNamedSampling(SeededTest):
    def test_shared_named(self):
        G_var = shared(value=np.atleast_2d(1.0), broadcastable=(True, False), name="G")

        with pm.Model():
            theta0 = pm.Normal(
                "theta0",
                mu=np.atleast_2d(0),
                tau=np.atleast_2d(1e20),
                shape=(1, 1),
                testval=np.atleast_2d(0),
            )
            theta = pm.Normal(
                "theta", mu=tt.dot(G_var, theta0), tau=np.atleast_2d(1e20), shape=(1, 1)
            )
            res = theta.random()
            assert np.isclose(res, 0.0)

    def test_shared_unnamed(self):
        G_var = shared(value=np.atleast_2d(1.0), broadcastable=(True, False))
        with pm.Model():
            theta0 = pm.Normal(
                "theta0",
                mu=np.atleast_2d(0),
                tau=np.atleast_2d(1e20),
                shape=(1, 1),
                testval=np.atleast_2d(0),
            )
            theta = pm.Normal(
                "theta", mu=tt.dot(G_var, theta0), tau=np.atleast_2d(1e20), shape=(1, 1)
            )
            res = theta.random()
            assert np.isclose(res, 0.0)

    def test_constant_named(self):
        G_var = tt.constant(np.atleast_2d(1.0), name="G")
        with pm.Model():
            theta0 = pm.Normal(
                "theta0",
                mu=np.atleast_2d(0),
                tau=np.atleast_2d(1e20),
                shape=(1, 1),
                testval=np.atleast_2d(0),
            )
            theta = pm.Normal(
                "theta", mu=tt.dot(G_var, theta0), tau=np.atleast_2d(1e20), shape=(1, 1)
            )

            res = theta.random()
            assert np.isclose(res, 0.0)


class TestChooseBackend:
    def test_choose_backend_none(self):
        with mock.patch("pymc3.sampling.NDArray") as nd:
            pm.sampling._choose_backend(None, "chain")
        assert nd.called

    def test_choose_backend_list_of_variables(self):
        with mock.patch("pymc3.sampling.NDArray") as nd:
            pm.sampling._choose_backend(["var1", "var2"], "chain")
        nd.assert_called_with(vars=["var1", "var2"])


class TestSamplePPC(SeededTest):
    def test_normal_scalar(self):
        nchains = 2
        ndraws = 500
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=0.0)
            trace = pm.sample(draws=ndraws, chains=nchains, return_inferencedata=False)

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive([model.test_point], samples=10)
            ppc0 = pm.fast_sample_posterior_predictive([model.test_point], samples=10)
            # deprecated argument is not introduced to fast version [2019/08/20:rpg]
            ppc = pm.sample_posterior_predictive(trace, var_names=["a"])
            # test empty ppc
            ppc = pm.sample_posterior_predictive(trace, var_names=[])
            assert len(ppc) == 0
            ppc = pm.fast_sample_posterior_predictive(trace, var_names=[])
            assert len(ppc) == 0

            # test keep_size parameter
            ppc = pm.sample_posterior_predictive(trace, keep_size=True)
            assert ppc["a"].shape == (nchains, ndraws)
            ppc = pm.fast_sample_posterior_predictive(trace, keep_size=True)
            assert ppc["a"].shape == (nchains, ndraws)

            # test keep_size parameter and idata input
            idata = az.from_pymc3(trace)
            ppc = pm.sample_posterior_predictive(idata, keep_size=True)
            assert ppc["a"].shape == (nchains, ndraws)
            ppc = pm.fast_sample_posterior_predictive(trace, keep_size=True)
            assert ppc["a"].shape == (nchains, ndraws)

            # test default case
            ppc = pm.sample_posterior_predictive(trace, var_names=["a"])
            assert "a" in ppc
            assert ppc["a"].shape == (nchains * ndraws,)
            # mu's standard deviation may have changed thanks to a's observed
            _, pval = stats.kstest(ppc["a"] - trace["mu"], stats.norm(loc=0, scale=1).cdf)
            assert pval > 0.001

            # test default case
            ppc = pm.fast_sample_posterior_predictive(trace, var_names=["a"])
            assert "a" in ppc
            assert ppc["a"].shape == (nchains * ndraws,)
            # mu's standard deviation may have changed thanks to a's observed
            _, pval = stats.kstest(ppc["a"] - trace["mu"], stats.norm(loc=0, scale=1).cdf)
            assert pval > 0.001

        # size argument not introduced to fast version [2019/08/20:rpg]
        with model:
            ppc = pm.sample_posterior_predictive(trace, size=5, var_names=["a"])
            assert ppc["a"].shape == (nchains * ndraws, 5)

    def test_normal_vector(self, caplog):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            trace = pm.sample(return_inferencedata=False)

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive([model.test_point], samples=10)
            ppc = pm.sample_posterior_predictive(trace, samples=12, var_names=[])
            assert len(ppc) == 0

            # test list input
            ppc0 = pm.fast_sample_posterior_predictive([model.test_point], samples=10)
            ppc = pm.fast_sample_posterior_predictive(trace, samples=12, var_names=[])
            assert len(ppc) == 0

            # test keep_size parameter
            ppc = pm.sample_posterior_predictive(trace, keep_size=True)
            assert ppc["a"].shape == (trace.nchains, len(trace), 2)
            with pytest.warns(UserWarning):
                ppc = pm.sample_posterior_predictive(trace, samples=12, var_names=["a"])
            assert "a" in ppc
            assert ppc["a"].shape == (12, 2)

            # test keep_size parameter with inference data as input...
            idata = az.from_pymc3(trace)
            ppc = pm.sample_posterior_predictive(idata, keep_size=True)
            assert ppc["a"].shape == (trace.nchains, len(trace), 2)
            with pytest.warns(UserWarning):
                ppc = pm.sample_posterior_predictive(trace, samples=12, var_names=["a"])
            assert "a" in ppc
            assert ppc["a"].shape == (12, 2)

            # test keep_size parameter
            ppc = pm.fast_sample_posterior_predictive(trace, keep_size=True)
            assert ppc["a"].shape == (trace.nchains, len(trace), 2)
            with pytest.warns(UserWarning):
                ppc = pm.fast_sample_posterior_predictive(trace, samples=12, var_names=["a"])
            assert "a" in ppc
            assert ppc["a"].shape == (12, 2)

            # test keep_size parameter with inference data as input
            ppc = pm.fast_sample_posterior_predictive(idata, keep_size=True)
            assert ppc["a"].shape == (trace.nchains, len(trace), 2)
            with pytest.warns(UserWarning):
                ppc = pm.fast_sample_posterior_predictive(trace, samples=12, var_names=["a"])
            assert "a" in ppc
            assert ppc["a"].shape == (12, 2)

            # size unsupported by fast_ version  argument. [2019/08/19:rpg]
            ppc = pm.sample_posterior_predictive(trace, samples=10, var_names=["a"], size=4)
            assert "a" in ppc
            assert ppc["a"].shape == (10, 4, 2)

    def test_exceptions(self, caplog):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            trace = pm.sample(return_inferencedata=False)

        with model:
            with pytest.raises(IncorrectArgumentsError):
                ppc = pm.sample_posterior_predictive(trace, samples=10, keep_size=True)
            with pytest.raises(IncorrectArgumentsError):
                ppc = pm.fast_sample_posterior_predictive(trace, samples=10, keep_size=True)

            # Not for fast_sample_posterior_predictive
            with pytest.raises(IncorrectArgumentsError):
                ppc = pm.sample_posterior_predictive(trace, size=4, keep_size=True)
            # test wrong type argument
            bad_trace = {"mu": stats.norm.rvs(size=1000)}
            with pytest.raises(TypeError):
                ppc = pm.sample_posterior_predictive(bad_trace)
            with pytest.raises(TypeError):
                ppc = pm.fast_sample_posterior_predictive(bad_trace)

    def test_vector_observed(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=1)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.0, 1.0]))
            trace = pm.sample(return_inferencedata=False)

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive([model.test_point], samples=10)
            ppc = pm.sample_posterior_predictive(trace, samples=12, var_names=[])
            assert len(ppc) == 0
            ppc = pm.sample_posterior_predictive(trace, samples=12, var_names=["a"])
            assert "a" in ppc
            assert ppc["a"].shape == (12, 2)

            ppc = pm.sample_posterior_predictive(trace, samples=10, var_names=["a"], size=4)
            assert "a" in ppc
            assert ppc["a"].shape == (10, 4, 2)

            # now with fast version
            # test list input
            ppc0 = pm.fast_sample_posterior_predictive([model.test_point], samples=10)
            ppc = pm.fast_sample_posterior_predictive(trace, samples=12, var_names=[])
            assert len(ppc) == 0
            ppc = pm.fast_sample_posterior_predictive(trace, samples=12, var_names=["a"])
            assert "a" in ppc
            assert ppc["a"].shape == (12, 2)

    def test_sum_normal(self):
        with pm.Model() as model:
            a = pm.Normal("a", sigma=0.2)
            b = pm.Normal("b", mu=a)
            trace = pm.sample(return_inferencedata=False)

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive([model.test_point], samples=10)
            assert ppc0 == {}
            ppc = pm.sample_posterior_predictive(trace, samples=1000, var_names=["b"])
            assert len(ppc) == 1
            assert ppc["b"].shape == (1000,)
            scale = np.sqrt(1 + 0.2 ** 2)
            _, pval = stats.kstest(ppc["b"], stats.norm(scale=scale).cdf)
            assert pval > 0.001

            # test list input
            ppc0 = pm.fast_sample_posterior_predictive([model.test_point], samples=10)
            assert ppc0 == {}
            ppc = pm.fast_sample_posterior_predictive(trace, samples=1000, var_names=["b"])
            assert len(ppc) == 1
            assert ppc["b"].shape == (1000,)
            scale = np.sqrt(1 + 0.2 ** 2)
            _, pval = stats.kstest(ppc["b"], stats.norm(scale=scale).cdf)
            assert pval > 0.001

    def test_model_not_drawable_prior(self):
        data = np.random.poisson(lam=10, size=200)
        model = pm.Model()
        with model:
            mu = pm.HalfFlat("sigma")
            pm.Poisson("foo", mu=mu, observed=data)
            trace = pm.sample(tune=1000, return_inferencedata=False)

        with model:
            with pytest.raises(ValueError) as excinfo:
                pm.sample_prior_predictive(50)
            assert "Cannot sample" in str(excinfo.value)
            samples = pm.sample_posterior_predictive(trace, 40)
            assert samples["foo"].shape == (40, 200)

            samples = pm.fast_sample_posterior_predictive(trace, 40)
            assert samples["foo"].shape == (40, 200)

    def test_model_shared_variable(self):
        x = np.random.randn(100)
        y = x > 0
        x_shared = theano.shared(x)
        y_shared = theano.shared(y)
        with pm.Model() as model:
            coeff = pm.Normal("x", mu=0, sd=1)
            logistic = pm.Deterministic("p", pm.math.sigmoid(coeff * x_shared))

            obs = pm.Bernoulli("obs", p=logistic, observed=y_shared)
            trace = pm.sample(100, return_inferencedata=False)

        x_shared.set_value([-1, 0, 1.0])
        y_shared.set_value([0, 0, 0])

        samples = 100
        with model:
            post_pred = pm.sample_posterior_predictive(
                trace, samples=samples, var_names=["p", "obs"]
            )

        expected_p = np.array([logistic.eval({coeff: val}) for val in trace["x"][:samples]])
        assert post_pred["obs"].shape == (samples, 3)
        npt.assert_allclose(post_pred["p"], expected_p)

        # fast version
        samples = 100
        with model:
            post_pred = pm.fast_sample_posterior_predictive(
                trace, samples=samples, var_names=["p", "obs"]
            )

        expected_p = np.array([logistic.eval({coeff: val}) for val in trace["x"][:samples]])
        assert post_pred["obs"].shape == (samples, 3)
        npt.assert_allclose(post_pred["p"], expected_p)

    def test_deterministic_of_observed(self):
        meas_in_1 = pm.theanof.floatX(2 + 4 * np.random.randn(10))
        meas_in_2 = pm.theanof.floatX(5 + 4 * np.random.randn(10))
        nchains = 2
        with pm.Model() as model:
            mu_in_1 = pm.Normal("mu_in_1", 0, 1)
            sigma_in_1 = pm.HalfNormal("sd_in_1", 1)
            mu_in_2 = pm.Normal("mu_in_2", 0, 1)
            sigma_in_2 = pm.HalfNormal("sd__in_2", 1)

            in_1 = pm.Normal("in_1", mu_in_1, sigma_in_1, observed=meas_in_1)
            in_2 = pm.Normal("in_2", mu_in_2, sigma_in_2, observed=meas_in_2)
            out_diff = in_1 + in_2
            pm.Deterministic("out", out_diff)

            trace = pm.sample(100, chains=nchains, return_inferencedata=False)
            np.random.seed(0)
            rtol = 1e-5 if theano.config.floatX == "float64" else 1e-4

            np.random.seed(0)
            ppc = pm.sample_posterior_predictive(
                model=model,
                trace=trace,
                samples=len(trace) * nchains,
                var_names=[var.name for var in (model.deterministics + model.basic_RVs)],
            )

            npt.assert_allclose(ppc["in_1"] + ppc["in_2"], ppc["out"], rtol=rtol)

            np.random.seed(0)
            ppc = pm.fast_sample_posterior_predictive(
                model=model,
                trace=trace,
                samples=len(trace) * nchains,
                var_names=[var.name for var in (model.deterministics + model.basic_RVs)],
            )

            npt.assert_allclose(ppc["in_1"] + ppc["in_2"], ppc["out"], rtol=rtol)

    def test_deterministic_of_observed_modified_interface(self):
        meas_in_1 = pm.theanof.floatX(2 + 4 * np.random.randn(100))
        meas_in_2 = pm.theanof.floatX(5 + 4 * np.random.randn(100))
        with pm.Model() as model:
            mu_in_1 = pm.Normal("mu_in_1", 0, 1)
            sigma_in_1 = pm.HalfNormal("sd_in_1", 1)
            mu_in_2 = pm.Normal("mu_in_2", 0, 1)
            sigma_in_2 = pm.HalfNormal("sd__in_2", 1)

            in_1 = pm.Normal("in_1", mu_in_1, sigma_in_1, observed=meas_in_1)
            in_2 = pm.Normal("in_2", mu_in_2, sigma_in_2, observed=meas_in_2)
            out_diff = in_1 + in_2
            pm.Deterministic("out", out_diff)

            trace = pm.sample(100, return_inferencedata=False)
            ppc_trace = pm.trace_to_dataframe(
                trace, varnames=[n for n in trace.varnames if n != "out"]
            ).to_dict("records")
            ppc = pm.sample_posterior_predictive(
                model=model,
                trace=ppc_trace,
                samples=len(ppc_trace),
                var_names=[x.name for x in (model.deterministics + model.basic_RVs)],
            )

            rtol = 1e-5 if theano.config.floatX == "float64" else 1e-3
            npt.assert_allclose(ppc["in_1"] + ppc["in_2"], ppc["out"], rtol=rtol)

            ppc = pm.fast_sample_posterior_predictive(
                model=model,
                trace=ppc_trace,
                samples=len(ppc_trace),
                var_names=[x.name for x in (model.deterministics + model.basic_RVs)],
            )

            rtol = 1e-5 if theano.config.floatX == "float64" else 1e-3
            npt.assert_allclose(ppc["in_1"] + ppc["in_2"], ppc["out"], rtol=rtol)

    def test_variable_type(self):
        with pm.Model() as model:
            mu = pm.HalfNormal("mu", 1)
            a = pm.Normal("a", mu=mu, sigma=2, observed=np.array([1, 2]))
            b = pm.Poisson("b", mu, observed=np.array([1, 2]))
            trace = pm.sample(return_inferencedata=False)

        with model:
            ppc = pm.sample_posterior_predictive(trace, samples=1)
            assert ppc["a"].dtype.kind == "f"
            assert ppc["b"].dtype.kind == "i"

    def test_potentials_warning(self):
        warning_msg = "The effect of Potentials on other parameters is ignored during"
        with pm.Model() as m:
            a = pm.Normal("a", 0, 1)
            p = pm.Potential("p", a + 1)
            obs = pm.Normal("obs", a, 1, observed=5)

        trace = az.from_dict({"a": np.random.rand(10)})
        with m:
            with pytest.warns(UserWarning, match=warning_msg):
                pm.sample_posterior_predictive(trace, samples=5)

            with pytest.warns(UserWarning, match=warning_msg):
                pm.fast_sample_posterior_predictive(trace, samples=5)


class TestSamplePPCW(SeededTest):
    def test_sample_posterior_predictive_w(self):
        data0 = np.random.normal(0, 1, size=50)
        warning_msg = "The number of samples is too small to check convergence reliably"

        with pm.Model() as model_0:
            mu = pm.Normal("mu", mu=0, sigma=1)
            y = pm.Normal("y", mu=mu, sigma=1, observed=data0)
            with pytest.warns(UserWarning, match=warning_msg):
                trace_0 = pm.sample(10, tune=0, chains=2, return_inferencedata=False)
            idata_0 = az.from_pymc3(trace_0)

        with pm.Model() as model_1:
            mu = pm.Normal("mu", mu=0, sigma=1, shape=len(data0))
            y = pm.Normal("y", mu=mu, sigma=1, observed=data0)
            with pytest.warns(UserWarning, match=warning_msg):
                trace_1 = pm.sample(10, tune=0, chains=2, return_inferencedata=False)
            idata_1 = az.from_pymc3(trace_1)

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

        trace = az.from_dict({"a": np.random.rand(10)})
        with pytest.warns(UserWarning, match=warning_msg):
            pm.sample_posterior_predictive_w(samples=5, traces=[trace, trace], models=[m, m])


@pytest.mark.parametrize(
    "method",
    [
        "jitter+adapt_diag",
        "adapt_diag",
        "advi",
        "ADVI+adapt_diag",
        "advi+adapt_diag_grad",
        "map",
        "advi_map",
        "adapt_full",
        "jitter+adapt_full",
    ],
)
def test_exec_nuts_init(method):
    with pm.Model() as model:
        pm.Normal("a", mu=0, sigma=1, shape=2)
        pm.HalfNormal("b", sigma=1)
    with model:
        start, _ = pm.init_nuts(init=method, n_init=10)
        assert isinstance(start, list)
        assert len(start) == 1
        assert isinstance(start[0], dict)
        assert "a" in start[0] and "b_log__" in start[0]
        start, _ = pm.init_nuts(init=method, n_init=10, chains=2)
        assert isinstance(start, list)
        assert len(start) == 2
        assert isinstance(start[0], dict)
        assert "a" in start[0] and "b_log__" in start[0]


@pytest.mark.parametrize(
    "init, start, expectation",
    [
        ("auto", None, pytest.raises(SamplingError)),
        ("jitter+adapt_diag", None, pytest.raises(SamplingError)),
        ("auto", {"x": 0}, does_not_raise()),
        ("jitter+adapt_diag", {"x": 0}, does_not_raise()),
        ("adapt_diag", None, does_not_raise()),
    ],
)
def test_default_sample_nuts_jitter(init, start, expectation, monkeypatch):
    # This test tries to check whether the starting points returned by init_nuts are actually
    # being used when pm.sample() is called without specifying an explicit start point (see
    # https://github.com/pymc-devs/pymc3/pull/4285).
    def _mocked_init_nuts(*args, **kwargs):
        if init == "adapt_diag":
            start_ = [{"x": np.array(0.79788456)}]
        else:
            start_ = [{"x": np.array(-0.04949886)}]
        _, step = pm.init_nuts(*args, **kwargs)
        return start_, step

    monkeypatch.setattr("pymc3.sampling.init_nuts", _mocked_init_nuts)
    with pm.Model() as m:
        x = pm.HalfNormal("x", transform=None)
        with expectation:
            pm.sample(tune=1, draws=0, chains=1, init=init, start=start, return_inferencedata=False)


@pytest.mark.parametrize(
    "testval, jitter_max_retries, expectation",
    [
        (0, 0, pytest.raises(SamplingError)),
        (0, 1, pytest.raises(SamplingError)),
        (0, 4, does_not_raise()),
        (0, 10, does_not_raise()),
        (1, 0, does_not_raise()),
    ],
)
def test_init_jitter(testval, jitter_max_retries, expectation):
    with pm.Model() as m:
        pm.HalfNormal("x", transform=None, testval=testval)

    with expectation:
        # Starting value is negative (invalid) when np.random.rand returns 0 (jitter = -1)
        # and positive (valid) when it returns 1 (jitter = 1)
        with mock.patch("numpy.random.rand", side_effect=[0, 0, 0, 1, 0]):
            start = pm.sampling._init_jitter(m, chains=1, jitter_max_retries=jitter_max_retries)
            pm.util.check_start_vals(start, m)


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
            observed_data = pm.Data("observed_data", observed)
            mu = pm.Normal("mu", mu=-100, sigma=1)
            positive_mu = pm.Deterministic("positive_mu", np.abs(mu))
            z = -1 - positive_mu
            pm.Normal("x_obs", mu=z, sigma=1, observed=observed_data)
            prior = pm.sample_prior_predictive()

        assert "observed_data" not in prior
        assert (prior["mu"] < 90).all()
        assert (prior["positive_mu"] > 90).all()
        assert (prior["x_obs"] < 90).all()
        assert prior["x_obs"].shape == (500, 200)
        npt.assert_array_almost_equal(prior["positive_mu"], np.abs(prior["mu"]), decimal=4)

    def test_respects_shape(self):
        for shape in (2, (2,), (10, 2), (10, 10)):
            with pm.Model():
                mu = pm.Gamma("mu", 3, 1, shape=1)
                goals = pm.Poisson("goals", mu, shape=shape)
                trace1 = pm.sample_prior_predictive(10, var_names=["mu", "mu", "goals"])
                trace2 = pm.sample_prior_predictive(10, var_names=["mu", "goals"])
            if shape == 2:  # want to test shape as an int
                shape = (2,)
            assert trace1["goals"].shape == (10,) + shape
            assert trace2["goals"].shape == (10,) + shape

    def test_multivariate(self):
        with pm.Model():
            m = pm.Multinomial("m", n=5, p=np.array([0.25, 0.25, 0.25, 0.25]), shape=4)
            trace = pm.sample_prior_predictive(10)

        assert m.random(size=10).shape == (10, 4)
        assert trace["m"].shape == (10, 4)

    def test_multivariate2(self):
        # Added test for issue #3271
        mn_data = np.random.multinomial(n=100, pvals=[1 / 6.0] * 6, size=10)
        with pm.Model() as dm_model:
            probs = pm.Dirichlet("probs", a=np.ones(6), shape=6)
            obs = pm.Multinomial("obs", n=100, p=probs, observed=mn_data)
            burned_trace = pm.sample(20, tune=10, cores=1, return_inferencedata=False)
        sim_priors = pm.sample_prior_predictive(samples=20, model=dm_model)
        sim_ppc = pm.sample_posterior_predictive(burned_trace, samples=20, model=dm_model)
        assert sim_priors["probs"].shape == (20, 6)
        assert sim_priors["obs"].shape == (20,) + obs.distribution.shape
        assert sim_ppc["obs"].shape == (20,) + obs.distribution.shape

        sim_ppc = pm.fast_sample_posterior_predictive(burned_trace, samples=20, model=dm_model)
        assert sim_ppc["obs"].shape == (20,) + obs.distribution.shape

    def test_layers(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0, upper=1, shape=10)
            b = pm.Binomial("b", n=1, p=a, shape=10)

        avg = b.random(size=10000).mean(axis=0)
        npt.assert_array_almost_equal(avg, 0.5 * np.ones_like(b), decimal=2)

    def test_transformed(self):
        n = 18
        at_bats = 45 * np.ones(n, dtype=int)
        hits = np.random.randint(1, 40, size=n, dtype=int)
        draws = 50

        with pm.Model() as model:
            phi = pm.Beta("phi", alpha=1.0, beta=1.0)

            kappa_log = pm.Exponential("logkappa", lam=5.0)
            kappa = pm.Deterministic("kappa", tt.exp(kappa_log))

            thetas = pm.Beta("thetas", alpha=phi * kappa, beta=(1.0 - phi) * kappa, shape=n)

            y = pm.Binomial("y", n=at_bats, p=thetas, observed=hits)
            gen = pm.sample_prior_predictive(draws)

        assert gen["phi"].shape == (draws,)
        assert gen["y"].shape == (draws, n)
        assert "thetas_logodds__" in gen

    def test_shared(self):
        n1 = 10
        obs = shared(np.random.rand(n1) < 0.5)
        draws = 50

        with pm.Model() as m:
            p = pm.Beta("p", 1.0, 1.0)
            y = pm.Bernoulli("y", p, observed=obs)
            gen1 = pm.sample_prior_predictive(draws)

        assert gen1["y"].shape == (draws, n1)

        n2 = 20
        obs.set_value(np.random.rand(n2) < 0.5)
        with m:
            gen2 = pm.sample_prior_predictive(draws)

        assert gen2["y"].shape == (draws, n2)

    def test_density_dist(self):
        obs = np.random.normal(-1, 0.1, size=10)
        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            sd = pm.Gamma("sd", 1, 2)
            a = pm.DensityDist(
                "a",
                pm.Normal.dist(mu, sd).logp,
                random=pm.Normal.dist(mu, sd).random,
                observed=obs,
            )
            prior = pm.sample_prior_predictive()

        npt.assert_almost_equal(prior["a"].mean(), 0, decimal=1)

    def test_shape_edgecase(self):
        with pm.Model():
            mu = pm.Normal("mu", shape=5)
            sd = pm.Uniform("sd", lower=2, upper=3)
            x = pm.Normal("x", mu=mu, sigma=sd, shape=5)
            prior = pm.sample_prior_predictive(10)
        assert prior["mu"].shape == (10, 5)

    def test_zeroinflatedpoisson(self):
        with pm.Model():
            theta = pm.Beta("theta", alpha=1, beta=1)
            psi = pm.HalfNormal("psi", sd=1)
            pm.ZeroInflatedPoisson("suppliers", psi=psi, theta=theta, shape=20)
            gen_data = pm.sample_prior_predictive(samples=5000)
            assert gen_data["theta"].shape == (5000,)
            assert gen_data["psi"].shape == (5000,)
            assert gen_data["suppliers"].shape == (5000, 20)

    def test_bounded_dist(self):
        with pm.Model() as model:
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
            x = BoundedNormal("x", mu=tt.zeros((3, 1)), sd=1 * tt.ones((3, 1)), shape=(3, 1))

        with model:
            prior_trace = pm.sample_prior_predictive(5)
            assert prior_trace["x"].shape == (5, 3, 1)

    def test_potentials_warning(self):
        warning_msg = "The effect of Potentials on other parameters is ignored during"
        with pm.Model() as m:
            a = pm.Normal("a", 0, 1)
            p = pm.Potential("p", a + 1)

        with m:
            with pytest.warns(UserWarning, match=warning_msg):
                pm.sample_prior_predictive(samples=5)


class TestSamplePosteriorPredictive:
    def test_point_list_arg_bug_fspp(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture
        with pmodel:
            pp = pm.fast_sample_posterior_predictive([trace[15]], var_names=["d"])

    def test_point_list_arg_bug_spp(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture
        with pmodel:
            pp = pm.sample_posterior_predictive([trace[15]], var_names=["d"])

    def test_sample_from_xarray_prior(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture

        with pmodel:
            prior = pm.sample_prior_predictive(samples=20)
        idat = az.from_pymc3(trace, prior=prior)
        with pmodel:
            pp = pm.sample_posterior_predictive(idat.prior, var_names=["d"])

    def test_sample_from_xarray_posterior(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture
        idat = az.from_pymc3(trace)
        with pmodel:
            pp = pm.sample_posterior_predictive(idat.posterior, var_names=["d"])

    def test_sample_from_xarray_posterior_fast(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture
        idat = az.from_pymc3(trace)
        with pmodel:
            pp = pm.fast_sample_posterior_predictive(idat.posterior, var_names=["d"])
