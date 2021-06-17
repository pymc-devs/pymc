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

import time

import aesara
import aesara.tensor as at
import numpy as np
import pytest

from aesara.graph.basic import ancestors
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.sort import SortOp
from arviz.data.inference_data import InferenceData

import pymc3 as pm

from pymc3.backends.base import MultiTrace
from pymc3.tests.helpers import SeededTest


class TestSMC(SeededTest):
    def setup_class(self):
        super().setup_class()
        self.samples = 1000
        n = 4
        mu1 = np.ones(n) * (1.0 / 2)
        mu2 = -mu1

        stdev = 0.1
        sigma = np.power(stdev, 2) * np.eye(n)
        isigma = np.linalg.inv(sigma)
        dsigma = np.linalg.det(sigma)

        w1 = stdev
        w2 = 1 - stdev

        def two_gaussians(x):
            log_like1 = (
                -0.5 * n * at.log(2 * np.pi)
                - 0.5 * at.log(dsigma)
                - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
            )
            log_like2 = (
                -0.5 * n * at.log(2 * np.pi)
                - 0.5 * at.log(dsigma)
                - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
            )
            return at.log(w1 * at.exp(log_like1) + w2 * at.exp(log_like2))

        with pm.Model() as self.SMC_test:
            X = pm.Uniform("X", lower=-2, upper=2.0, shape=n)
            llk = pm.Potential("muh", two_gaussians(X))

        self.muref = mu1

        with pm.Model() as self.fast_model:
            x = pm.Normal("x", 0, 1)
            y = pm.Normal("y", x, 1, observed=0)

        with pm.Model() as self.slow_model:
            x = pm.Normal("x", 0, 1)
            y = pm.Normal("y", x, 1, observed=100)

    def test_sample(self):
        with self.SMC_test:

            mtrace = pm.sample_smc(
                draws=self.samples,
                cores=1,  # Fails in parallel due to #4799
                return_inferencedata=False,
            )

        x = mtrace["X"]
        mu1d = np.abs(x).mean(axis=0)
        np.testing.assert_allclose(self.muref, mu1d, rtol=0.0, atol=0.03)

    def test_discrete_continuous(self):
        with pm.Model() as model:
            a = pm.Poisson("a", 5)
            b = pm.HalfNormal("b", 10)
            y = pm.Normal("y", a, b, observed=[1, 2, 3, 4])
            trace = pm.sample_smc(draws=10)

    def test_ml(self):
        data = np.repeat([1, 0], [50, 50])
        marginals = []
        a_prior_0, b_prior_0 = 1.0, 1.0
        a_prior_1, b_prior_1 = 20.0, 20.0

        for alpha, beta in ((a_prior_0, b_prior_0), (a_prior_1, b_prior_1)):
            with pm.Model() as model:
                a = pm.Beta("a", alpha, beta)
                y = pm.Bernoulli("y", a, observed=data)
                trace = pm.sample_smc(2000, return_inferencedata=False)
                marginals.append(trace.report.log_marginal_likelihood)
        # compare to the analytical result
        assert abs(np.exp(np.mean(marginals[1]) - np.mean(marginals[0])) - 4.0) <= 1

    def test_start(self):
        with pm.Model() as model:
            a = pm.Poisson("a", 5)
            b = pm.HalfNormal("b", 10)
            y = pm.Normal("y", a, b, observed=[1, 2, 3, 4])
            start = {
                "a": np.random.poisson(5, size=500),
                "b_log__": np.abs(np.random.normal(0, 10, size=500)),
            }
            trace = pm.sample_smc(500, chains=1, start=start)

    def test_slowdown_warning(self):
        with aesara.config.change_flags(floatX="float32"):
            with pytest.warns(UserWarning, match="SMC sampling may run slower due to"):
                with pm.Model() as model:
                    a = pm.Poisson("a", 5)
                    y = pm.Normal("y", a, 5, observed=[1, 2, 3, 4])
                    trace = pm.sample_smc(draws=100, chains=2, cores=1)

    @pytest.mark.parametrize("chains", (1, 2))
    def test_return_datatype(self, chains):
        draws = 10

        with self.fast_model:
            idata = pm.sample_smc(chains=chains, draws=draws)
            mt = pm.sample_smc(chains=chains, draws=draws, return_inferencedata=False)

        assert isinstance(idata, InferenceData)
        assert "sample_stats" in idata
        assert idata.posterior.dims["chain"] == chains
        assert idata.posterior.dims["draw"] == draws

        assert isinstance(mt, MultiTrace)
        assert mt.nchains == chains
        assert mt["x"].size == chains * draws

    def test_convergence_checks(self):
        with self.fast_model:
            with pytest.warns(
                UserWarning,
                match="The number of samples is too small",
            ):
                pm.sample_smc(draws=99)

    def test_parallel_sampling(self):
        # Cache graph
        with self.slow_model:
            _ = pm.sample_smc(draws=10, chains=1, cores=1, return_inferencedata=False)

        chains = 4
        draws = 100

        t0 = time.time()
        with self.slow_model:
            idata = pm.sample_smc(draws=draws, chains=chains, cores=4)
        t_mp = time.time() - t0
        assert idata.posterior.dims["chain"] == chains
        assert idata.posterior.dims["draw"] == draws

        t0 = time.time()
        with self.slow_model:
            idata = pm.sample_smc(draws=draws, chains=chains, cores=1)
        t_seq = time.time() - t0
        assert idata.posterior.dims["chain"] == chains
        assert idata.posterior.dims["draw"] == draws

        assert t_mp < t_seq

    def test_depracated_parallel_arg(self):
        with self.fast_model:
            with pytest.warns(
                DeprecationWarning,
                match="The argument parallel is deprecated",
            ):
                pm.sample_smc(draws=10, chains=1, parallel=False)


def normal_sim(rng, a, b, size):
    return rng.normal(a, b, size=size)


class NormalSimBaseRV(pm.SimulatorRV):
    ndim_supp = 0
    ndims_params = [0, 0]
    fn = normal_sim


class NormalSimRV1(NormalSimBaseRV):
    distance = "gaussian"
    sum_stat = "sort"


class NormalSimRV2(NormalSimBaseRV):
    distance = "laplace"
    sum_stat = "mean"
    epsilon = 0.1


class NormalSimRV3(NormalSimBaseRV):
    distance = "gaussian"
    sum_stat = "mean"
    epsilon = 0.5


def abs_diff(eps, obs_data, sim_data):
    return np.mean(np.abs((obs_data - sim_data) / eps))


def quantiles(x):
    return np.quantile(x, [0.25, 0.5, 0.75])


class NormalSimCustomRV1(NormalSimBaseRV):
    distance = abs_diff
    sum_stat = quantiles


class NormalSimCustomRV2(NormalSimBaseRV):
    distance = abs_diff
    sum_stat = "mean"


class TestSimulator(SeededTest):
    """
    Tests for pm.Simulator. They are included in this file because Simulator was
    designed primarily to be used with SMC sampling.
    """

    @staticmethod
    def count_rvs(end_node):
        return len(
            [
                node
                for node in ancestors([end_node])
                if node.owner is not None and isinstance(node.owner.op, RandomVariable)
            ]
        )

    def setup_class(self):
        super().setup_class()
        self.data = np.random.normal(loc=0, scale=1, size=1000)

        with pm.Model() as self.SMABC_test:
            a = pm.Normal("a", mu=0, sigma=1)
            b = pm.HalfNormal("b", sigma=1)
            s = pm.Simulator("s", NormalSimRV1(), a, b, observed=self.data)
            self.s = s

        with pm.Model() as self.SMABC_potential:
            a = pm.Normal("a", mu=0, sigma=1, initval=0.5)
            b = pm.HalfNormal("b", sigma=1)
            c = pm.Potential("c", pm.math.switch(a > 0, 0, -np.inf))
            s = pm.Simulator("s", NormalSimRV1(), a, b, observed=self.data)

    def test_one_gaussian(self):
        assert self.count_rvs(self.SMABC_test.logpt) == 1

        with self.SMABC_test:
            trace = pm.sample_smc(draws=1000, return_inferencedata=False)
            pr_p = pm.sample_prior_predictive(1000)
            po_p = pm.sample_posterior_predictive(trace, 1000)

        assert abs(self.data.mean() - trace["a"].mean()) < 0.05
        assert abs(self.data.std() - trace["b"].mean()) < 0.05

        assert pr_p["s"].shape == (1000, 1000)
        assert abs(0 - pr_p["s"].mean()) < 0.10
        assert abs(1.4 - pr_p["s"].std()) < 0.10

        assert po_p["s"].shape == (1000, 1000)
        assert abs(self.data.mean() - po_p["s"].mean()) < 0.10
        assert abs(self.data.std() - po_p["s"].std()) < 0.10

    def test_custom_dist_sum_stat(self):
        with pm.Model() as m:
            a = pm.Normal("a", mu=0, sigma=1)
            b = pm.HalfNormal("b", sigma=1)
            s = pm.Simulator("s", NormalSimCustomRV1(), a, b, observed=self.data)

        assert self.count_rvs(m.logpt) == 1

        with m:
            pm.sample_smc(draws=100, chains=2)

    def test_custom_dist_sum_stat_scalar(self):
        """
        Test that automatically wrapped functions cope well with scalar inputs
        """
        scalar_data = 5

        with pm.Model() as m:
            s = pm.Simulator("s", NormalSimCustomRV1(), 0, 1, observed=scalar_data)
        assert self.count_rvs(m.logpt) == 1

        with pm.Model() as m:
            s = pm.Simulator("s", NormalSimCustomRV2(), 0, 1, observed=scalar_data)
        assert self.count_rvs(m.logpt) == 1

    def test_model_with_potential(self):
        assert self.count_rvs(self.SMABC_potential.logpt) == 1

        with self.SMABC_potential:
            trace = pm.sample_smc(draws=100, chains=1, return_inferencedata=False)
            assert np.all(trace["a"] >= 0)

    def test_simulator_metropolis_mcmc(self):
        with self.SMABC_test as m:
            step = pm.Metropolis([m.rvs_to_values[m["a"]], m.rvs_to_values[m["b"]]])
            trace = pm.sample(step=step, return_inferencedata=False)

        assert abs(self.data.mean() - trace["a"].mean()) < 0.05
        assert abs(self.data.std() - trace["b"].mean()) < 0.05

    def test_multiple_simulators(self):
        true_a = 2
        true_b = -2

        data1 = np.random.normal(true_a, 0.1, size=1000)
        data2 = np.random.normal(true_b, 0.1, size=1000)

        with pm.Model() as m:
            a = pm.Normal("a", mu=0, sigma=3)
            b = pm.Normal("b", mu=0, sigma=3)
            sim1 = pm.Simulator("sim1", NormalSimRV1(), a, 0.1, observed=data1)
            sim2 = pm.Simulator("sim2", NormalSimRV2(), b, 0.1, observed=data2)

        assert self.count_rvs(m.logpt) == 2

        # Check that the logps use the correct methods
        a_val = m.rvs_to_values[a]
        sim1_val = m.rvs_to_values[sim1]
        logp_sim1 = pm.logp(sim1, sim1_val)
        logp_sim1_fn = aesara.function([sim1_val, a_val], logp_sim1)

        b_val = m.rvs_to_values[b]
        sim2_val = m.rvs_to_values[sim2]
        logp_sim2 = pm.logp(sim2, sim2_val)
        logp_sim2_fn = aesara.function([sim2_val, b_val], logp_sim2)

        assert any(
            node for node in logp_sim1_fn.maker.fgraph.toposort() if isinstance(node.op, SortOp)
        )

        assert not any(
            node for node in logp_sim2_fn.maker.fgraph.toposort() if isinstance(node.op, SortOp)
        )

        with m:
            trace = pm.sample_smc(return_inferencedata=False)
        assert abs(true_a - trace["a"].mean()) < 0.05
        assert abs(true_b - trace["b"].mean()) < 0.05

    def test_nested_simulators(self):
        true_a = 2
        data = np.random.normal(true_a, 0.1, size=1000)

        with pm.Model() as m:
            sim1 = pm.Simulator("sim1", NormalSimRV3(), 0, 4)
            sim2 = pm.Simulator("sim2", NormalSimRV3(), sim1, 0.1, observed=data)

        assert self.count_rvs(m.logpt) == 2

        with m:
            trace = pm.sample_smc(return_inferencedata=False)

        assert (true_a - trace["sim1"].mean()) < 0.1

    def test_simulator_rv_error_msg(self):
        class RV(pm.SimulatorRV):
            pass

        msg = "SimulatorRV fn was not specified"
        with pytest.raises(ValueError, match=msg):
            RV()

        class RV(pm.SimulatorRV):
            fn = lambda: None

        msg = "SimulatorRV must specify `ndim_supp`"
        with pytest.raises(ValueError, match=msg):
            RV()

        class RV(pm.SimulatorRV):
            fn = lambda: None
            ndim_supp = 0

        msg = "Simulator RV must specify `ndims_params`"
        with pytest.raises(ValueError, match=msg):
            RV()

        class RV(pm.SimulatorRV):
            fn = lambda: None
            ndim_supp = 0
            ndims_params = [0, 0, 0]
            distance = "not_real"

        msg = "The distance metric not_real is not implemented"
        with pytest.raises(ValueError, match=msg):
            RV()

        class RV(pm.SimulatorRV):
            fn = lambda: None
            ndim_supp = 0
            ndims_params = [0, 0, 0]
            distance = "gaussian"
            sum_stat = "not_real"

        msg = "The summary statistic not_real is not implemented"
        with pytest.raises(ValueError, match=msg):
            RV()

    def test_simulator_error_msg(self):
        msg = "does not seem to be an instantiated"
        with pm.Model() as m:
            with pytest.raises(ValueError, match=msg):
                pm.Simulator("sim", NormalSimRV1, 0, 1)

        msg = "should be a subclass instance of"
        with pm.Model() as m:
            with pytest.raises(ValueError, match=msg):
                pm.Simulator("sim", lambda: None, 0, 1)

        msg = "`Simulator` expected 2 parameters"
        with pm.Model() as m:
            with pytest.raises(ValueError, match=msg):
                pm.Simulator("sim", NormalSimRV1(), 0)

        msg = "distance is no longer defined when calling `pm.Simulator`"
        with pm.Model() as m:
            with pytest.raises(ValueError, match=msg):
                pm.Simulator("sim", NormalSimRV1(), 0, 1, distance="gaussian")

        msg = "sum_stat is no longer defined when calling `pm.Simulator`"
        with pm.Model() as m:
            with pytest.raises(ValueError, match=msg):
                pm.Simulator("sim", NormalSimRV1(), 0, 1, sum_stat="sort")

        msg = "epsilon is no longer defined when calling `pm.Simulator`"
        with pm.Model() as m:
            with pytest.raises(ValueError, match=msg):
                pm.Simulator("sim", NormalSimRV1(), 0, 1, epsilon=1.0)

    def test_params_kwarg_deprecation(self):
        msg = "The kwarg ``params`` will be deprecated."
        with pm.Model() as m:
            with pytest.warns(DeprecationWarning, match=msg):
                pm.Simulator("sim", NormalSimRV1(), params=(0, 1))

    @pytest.mark.xfail(reason="KL not refactored")
    def test_automatic_use_of_sort(self):
        with pm.Model() as model:
            s_k = pm.Simulator(
                "s_k",
                None,
                params=None,
                distance="kullback_leibler",
                sum_stat="sort",
                observed=self.data,
            )
        assert s_k.distribution.sum_stat is pm.distributions.simulator.identity

    def test_name_is_string_type(self):
        with self.SMABC_potential:
            assert not self.SMABC_potential.name
            trace = pm.sample_smc(draws=10, cores=1, return_inferencedata=False)
            assert isinstance(trace._straces[0].name, str)

    def test_named_models_are_unsupported(self):
        with pm.Model(name="NamedModel"):
            a = pm.Normal("a", mu=0, sigma=1)
            b = pm.HalfNormal("b", sigma=1)
            s = pm.Simulator("s", NormalSimRV1(), a, b, observed=self.data)

            # TODO: Why is this?
            with pytest.raises(NotImplementedError, match="named models"):
                pm.sample_smc(draws=10, chains=1)
