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
import warnings

import aesara
import numpy as np
import pytest
import scipy.stats as st

from aesara.graph import ancestors
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.var import (
    RandomGeneratorSharedVariable,
    RandomStateSharedVariable,
)
from aesara.tensor.sort import SortOp

import pymc as pm

from pymc import floatX
from pymc.aesaraf import compile_pymc
from pymc.initial_point import make_initial_point_fn
from pymc.smc.kernels import IMH
from pymc.tests.helpers import SeededTest


class TestSimulator(SeededTest):
    @staticmethod
    def count_rvs(end_node):
        return len(
            [
                node
                for node in ancestors([end_node])
                if node.owner is not None and isinstance(node.owner.op, RandomVariable)
            ]
        )

    @staticmethod
    def normal_sim(rng, a, b, size):
        return rng.normal(a, b, size=size)

    @staticmethod
    def abs_diff(eps, obs_data, sim_data):
        return np.mean(np.abs((obs_data - sim_data) / eps))

    @staticmethod
    def quantiles(x):
        return np.quantile(x, [0.25, 0.5, 0.75])

    def setup_class(self):
        super().setup_class()
        self.data = np.random.normal(loc=0, scale=1, size=1000)

        with pm.Model() as self.SMABC_test:
            a = pm.Normal("a", mu=0, sigma=1)
            b = pm.HalfNormal("b", sigma=1)
            s = pm.Simulator("s", self.normal_sim, a, b, sum_stat="sort", observed=self.data)
            self.s = s

        with pm.Model() as self.SMABC_potential:
            a = pm.Normal("a", mu=0, sigma=1, initval=0.5)
            b = pm.HalfNormal("b", sigma=1)
            c = pm.Potential("c", pm.math.switch(a > 0, 0, -np.inf))
            s = pm.Simulator("s", self.normal_sim, a, b, observed=self.data)

    def test_one_gaussian(self):
        assert self.count_rvs(self.SMABC_test.logp()) == 1

        with self.SMABC_test:
            trace = pm.sample_smc(draws=1000, chains=1, return_inferencedata=False)
            pr_p = pm.sample_prior_predictive(1000, return_inferencedata=False)
            po_p = pm.sample_posterior_predictive(trace, return_inferencedata=False)

        assert abs(self.data.mean() - trace["a"].mean()) < 0.05
        assert abs(self.data.std() - trace["b"].mean()) < 0.05

        assert pr_p["s"].shape == (1000, 1000)
        assert abs(0 - pr_p["s"].mean()) < 0.15
        assert abs(1.4 - pr_p["s"].std()) < 0.10

        assert po_p["s"].shape == (1, 1000, 1000)
        assert abs(self.data.mean() - po_p["s"].mean()) < 0.10
        assert abs(self.data.std() - po_p["s"].std()) < 0.10

    @pytest.mark.parametrize("floatX", ["float32", "float64"])
    def test_custom_dist_sum_stat(self, floatX):
        with aesara.config.change_flags(floatX=floatX):
            with pm.Model() as m:
                a = pm.Normal("a", mu=0, sigma=1)
                b = pm.HalfNormal("b", sigma=1)
                s = pm.Simulator(
                    "s",
                    self.normal_sim,
                    a,
                    b,
                    distance=self.abs_diff,
                    sum_stat=self.quantiles,
                    observed=self.data,
                )

            assert self.count_rvs(m.logp()) == 1

            with m:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "More chains .* than draws .*", UserWarning)
                    pm.sample_smc(draws=100)

    @pytest.mark.parametrize("floatX", ["float32", "float64"])
    def test_custom_dist_sum_stat_scalar(self, floatX):
        """
        Test that automatically wrapped functions cope well with scalar inputs
        """
        scalar_data = 5

        with aesara.config.change_flags(floatX=floatX):
            with pm.Model() as m:
                s = pm.Simulator(
                    "s",
                    self.normal_sim,
                    0,
                    1,
                    distance=self.abs_diff,
                    sum_stat=self.quantiles,
                    observed=scalar_data,
                )
            assert self.count_rvs(m.logp()) == 1

            with pm.Model() as m:
                s = pm.Simulator(
                    "s",
                    self.normal_sim,
                    0,
                    1,
                    distance=self.abs_diff,
                    sum_stat="mean",
                    observed=scalar_data,
                )
            assert self.count_rvs(m.logp()) == 1

    def test_model_with_potential(self):
        assert self.count_rvs(self.SMABC_potential.logp()) == 1

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
            sim1 = pm.Simulator(
                "sim1",
                self.normal_sim,
                a,
                0.1,
                distance="gaussian",
                sum_stat="sort",
                observed=data1,
            )
            sim2 = pm.Simulator(
                "sim2",
                self.normal_sim,
                b,
                0.1,
                distance="laplace",
                sum_stat="mean",
                epsilon=0.1,
                observed=data2,
            )

        assert self.count_rvs(m.logp()) == 2

        # Check that the logps use the correct methods
        logp_sim1_fn = m.compile_fn(m.logp(sim1), point_fn=False)
        logp_sim2_fn = m.compile_fn(m.logp(sim2), point_fn=False)

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
        rng = self.get_random_state()
        data = rng.normal(true_a, 0.1, size=1000)

        with pm.Model() as m:
            sim1 = pm.Simulator(
                "sim1",
                self.normal_sim,
                params=(0, 4),
                distance="gaussian",
                sum_stat="identity",
            )
            sim2 = pm.Simulator(
                "sim2",
                self.normal_sim,
                params=(sim1, 0.1),
                distance="gaussian",
                sum_stat="mean",
                epsilon=0.1,
                observed=data,
            )

        assert self.count_rvs(m.logp()) == 2

        with m:
            trace = pm.sample_smc(return_inferencedata=False)

        assert np.abs(true_a - trace["sim1"].mean()) < 0.1

    def test_upstream_rngs_not_in_compiled_logp(self):
        smc = IMH(model=self.SMABC_test)
        smc.initialize_population()
        smc._initialize_kernel()
        likelihood_func = smc.likelihood_logp_func

        # Test graph is stochastic
        inarray = floatX(np.array([0, 0]))
        assert likelihood_func(inarray) != likelihood_func(inarray)

        # Test only one shared RNG is present
        compiled_graph = likelihood_func.maker.fgraph.outputs
        shared_rng_vars = [
            node
            for node in ancestors(compiled_graph)
            if isinstance(node, (RandomStateSharedVariable, RandomGeneratorSharedVariable))
        ]
        assert len(shared_rng_vars) == 1

    def test_simulator_error_msg(self):
        msg = "The distance metric not_real is not implemented"
        with pytest.raises(ValueError, match=msg):
            with pm.Model() as m:
                sim = pm.Simulator("sim", self.normal_sim, 0, 1, distance="not_real")

        msg = "The summary statistic not_real is not implemented"
        with pytest.raises(ValueError, match=msg):
            with pm.Model() as m:
                sim = pm.Simulator("sim", self.normal_sim, 0, 1, sum_stat="not_real")

        msg = "Cannot pass both unnamed parameters and `params`"
        with pytest.raises(ValueError, match=msg):
            with pm.Model() as m:
                sim = pm.Simulator("sim", self.normal_sim, 0, params=(1))

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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                warnings.filterwarnings(
                    "ignore", "invalid value encountered in true_divide", RuntimeWarning
                )
                trace = pm.sample_smc(draws=10, chains=1, return_inferencedata=False)
            assert isinstance(trace._straces[0].name, str)

    def test_named_model(self):
        # Named models used to fail with Simulator because the arguments to the
        # random fn used to be passed by name. This is no longer true.
        # https://github.com/pymc-devs/pymc/pull/4365#issuecomment-761221146
        name = "NamedModel"
        with pm.Model(name=name):
            a = pm.Normal("a", mu=0, sigma=1)
            b = pm.HalfNormal("b", sigma=1)
            s = pm.Simulator("s", self.normal_sim, a, b, observed=self.data)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                trace = pm.sample_smc(draws=10, chains=2, return_inferencedata=False)
            assert f"{name}::a" in trace.varnames
            assert f"{name}::b" in trace.varnames
            assert f"{name}::b_log__" in trace.varnames

    @pytest.mark.parametrize("mu", [0, np.arange(3)], ids=str)
    @pytest.mark.parametrize("sigma", [1, np.array([1, 2, 5])], ids=str)
    @pytest.mark.parametrize("size", [None, 3, (5, 3)], ids=str)
    def test_simulator_moment(self, mu, sigma, size):
        def normal_sim(rng, mu, sigma, size):
            return rng.normal(mu, sigma, size=size)

        with pm.Model() as model:
            x = pm.Simulator("x", normal_sim, mu, sigma, size=size)

        fn = make_initial_point_fn(
            model=model,
            return_transformed=False,
            default_strategy="moment",
        )

        random_draw = model["x"].eval()
        result = fn(0)["x"]
        assert result.shape == random_draw.shape

        # We perform a z-test between the moment and expected mean from a sample of 10 draws
        # This test fails if the number of samples averaged in moment(Simulator)
        # is much smaller than 10, but would not catch the case where the number of samples
        # is higher than the expected 10

        n = 10  # samples
        expected_sample_mean = mu
        expected_sample_mean_std = np.sqrt(sigma**2 / n)

        # Multiple test adjustment for z-test to maintain alpha=0.01
        alpha = 0.01
        alpha /= 2 * 2 * 3  # Correct for number of test permutations
        alpha /= random_draw.size  # Correct for distribution size
        cutoff = st.norm().ppf(1 - (alpha / 2))

        assert np.all(np.abs((result - expected_sample_mean) / expected_sample_mean_std) < cutoff)

    def test_dist(self):
        x = pm.Simulator.dist(self.normal_sim, 0, 1, sum_stat="sort", shape=(3,), class_name="test")
        x_logp = pm.logp(x, [0, 1, 2])

        x_logp_fn = compile_pymc([], x_logp, random_seed=1)
        res1, res2 = x_logp_fn(), x_logp_fn()
        assert res1.shape == (3,)
        assert np.all(res1 != res2)

        x_logp_fn = compile_pymc([], x_logp, random_seed=1)
        res3, res4 = x_logp_fn(), x_logp_fn()
        assert np.all(res1 == res3)
        assert np.all(res2 == res4)
