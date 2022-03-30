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

import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.stats as st

from aesara.graph.basic import ancestors
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.var import (
    RandomGeneratorSharedVariable,
    RandomStateSharedVariable,
)
from aesara.tensor.sort import SortOp
from arviz.data.inference_data import InferenceData

import pymc as pm

from pymc.aesaraf import floatX
from pymc.backends.base import MultiTrace
from pymc.smc.smc import IMH
from pymc.tests.helpers import SeededTest, assert_random_state_equal


class TestSMC(SeededTest):
    """Tests for the default SMC kernel"""

    def setup_class(self):
        super().setup_class()
        self.samples = 1000
        n = 4
        mu1 = np.ones(n) * 0.5
        mu2 = -mu1

        stdev = 0.1
        sigma = np.power(stdev, 2) * np.eye(n)
        isigma = np.linalg.inv(sigma)
        dsigma = np.linalg.det(sigma)

        w1 = stdev
        w2 = 1 - stdev

        def two_gaussians(x):
            """
            Mixture of gaussians likelihood
            """
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

    def test_sample(self):
        initial_rng_state = np.random.get_state()
        with self.SMC_test:
            mtrace = pm.sample_smc(draws=self.samples, return_inferencedata=False)

        # Verify sampling was done with a non-global random generator
        assert_random_state_equal(initial_rng_state, np.random.get_state())
        x = mtrace["X"]
        mu1d = np.abs(x).mean(axis=0)
        np.testing.assert_allclose(self.muref, mu1d, rtol=0.0, atol=0.03)

    def test_discrete_rounding_proposal(self):
        """
        Test that discrete variable values are automatically rounded
        in SMC logp functions
        """

        with pm.Model() as m:
            z = pm.Bernoulli("z", p=0.7)
            like = pm.Potential("like", z * 1.0)

        smc = IMH(model=m)
        smc.initialize_population()
        smc._initialize_kernel()

        assert smc.prior_logp_func(floatX(np.array([-0.51]))) == -np.inf
        assert np.isclose(smc.prior_logp_func(floatX(np.array([-0.49]))), np.log(0.3))
        assert np.isclose(smc.prior_logp_func(floatX(np.array([0.49]))), np.log(0.3))
        assert np.isclose(smc.prior_logp_func(floatX(np.array([0.51]))), np.log(0.7))
        assert smc.prior_logp_func(floatX(np.array([1.51]))) == -np.inf

    def test_unobserved_discrete(self):
        n = 10
        rng = self.get_random_state()
        z_true = np.zeros(n, dtype=int)
        z_true[int(n / 2) :] = 1
        y = st.norm(np.array([-1, 1])[z_true], 0.25).rvs(random_state=rng)

        with pm.Model() as m:
            z = pm.Bernoulli("z", p=0.5, size=n)
            mu = pm.math.switch(z, 1.0, -1.0)
            like = pm.Normal("like", mu=mu, sigma=0.25, observed=y)

            trace = pm.sample_smc(chains=1, return_inferencedata=False)

        assert np.all(np.median(trace["z"], axis=0) == z_true)

    def test_marginal_likelihood(self):
        """
        Verifies that the log marginal likelihood function
        can be correctly computed for a Beta-Bernoulli model.
        """
        data = np.repeat([1, 0], [50, 50])
        marginals = []
        a_prior_0, b_prior_0 = 1.0, 1.0
        a_prior_1, b_prior_1 = 20.0, 20.0

        for alpha, beta in ((a_prior_0, b_prior_0), (a_prior_1, b_prior_1)):
            with pm.Model() as model:
                a = pm.Beta("a", alpha, beta)
                y = pm.Bernoulli("y", a, observed=data)
                trace = pm.sample_smc(2000, chains=2, return_inferencedata=False)
            # log_marignal_likelihood is found in the last value of each chain
            lml = np.mean([chain[-1] for chain in trace.report.log_marginal_likelihood])
            marginals.append(lml)

        # compare to the analytical result
        assert abs(np.exp(marginals[1] - marginals[0]) - 4.0) <= 1

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

    def test_kernel_kwargs(self):
        with self.fast_model:
            trace = pm.sample_smc(
                draws=10,
                chains=1,
                threshold=0.7,
                correlation_threshold=0.02,
                return_inferencedata=False,
                kernel=pm.smc.IMH,
            )

            assert trace.report.threshold == 0.7
            assert trace.report.n_draws == 10

            assert trace.report.correlation_threshold == 0.02

        with self.fast_model:
            trace = pm.sample_smc(
                draws=10,
                chains=1,
                threshold=0.95,
                correlation_threshold=0.02,
                return_inferencedata=False,
                kernel=pm.smc.MH,
            )

            assert trace.report.threshold == 0.95
            assert trace.report.n_draws == 10
            assert trace.report.correlation_threshold == 0.02

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

    def test_deprecated_parallel_arg(self):
        with self.fast_model:
            with pytest.warns(
                FutureWarning,
                match="The argument parallel is deprecated",
            ):
                pm.sample_smc(draws=10, chains=1, parallel=False)

    def test_deprecated_abc_args(self):
        with self.fast_model:
            with pytest.warns(
                FutureWarning,
                match='The kernel string argument "ABC" in sample_smc has been deprecated',
            ):
                pm.sample_smc(draws=10, chains=1, kernel="ABC")

            with pytest.warns(
                FutureWarning,
                match='The kernel string argument "Metropolis" in sample_smc has been deprecated',
            ):
                pm.sample_smc(draws=10, chains=1, kernel="Metropolis")

            with pytest.warns(
                FutureWarning,
                match="save_sim_data has been deprecated",
            ):
                pm.sample_smc(draws=10, chains=1, save_sim_data=True)

            with pytest.warns(
                FutureWarning,
                match="save_log_pseudolikelihood has been deprecated",
            ):
                pm.sample_smc(draws=10, chains=1, save_log_pseudolikelihood=True)


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
        assert self.count_rvs(self.SMABC_test.logpt()) == 1

        with self.SMABC_test:
            trace = pm.sample_smc(draws=1000, chains=1, return_inferencedata=False)
            pr_p = pm.sample_prior_predictive(1000, return_inferencedata=False)
            po_p = pm.sample_posterior_predictive(
                trace, keep_size=False, return_inferencedata=False
            )

        assert abs(self.data.mean() - trace["a"].mean()) < 0.05
        assert abs(self.data.std() - trace["b"].mean()) < 0.05

        assert pr_p["s"].shape == (1000, 1000)
        assert abs(0 - pr_p["s"].mean()) < 0.15
        assert abs(1.4 - pr_p["s"].std()) < 0.10

        assert po_p["s"].shape == (1000, 1000)
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

            assert self.count_rvs(m.logpt()) == 1

            with m:
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
            assert self.count_rvs(m.logpt()) == 1

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
            assert self.count_rvs(m.logpt()) == 1

    def test_model_with_potential(self):
        assert self.count_rvs(self.SMABC_potential.logpt()) == 1

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

        assert self.count_rvs(m.logpt()) == 2

        # Check that the logps use the correct methods
        a_val = m.rvs_to_values[a]
        sim1_val = m.rvs_to_values[sim1]
        logp_sim1 = pm.joint_logpt(sim1, sim1_val)
        logp_sim1_fn = aesara.function([a_val], logp_sim1)

        b_val = m.rvs_to_values[b]
        sim2_val = m.rvs_to_values[sim2]
        logp_sim2 = pm.joint_logpt(sim2, sim2_val)
        logp_sim2_fn = aesara.function([b_val], logp_sim2)

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

        assert self.count_rvs(m.logpt()) == 2

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

            trace = pm.sample_smc(draws=10, chains=2, return_inferencedata=False)
            assert f"{name}::a" in trace.varnames
            assert f"{name}::b" in trace.varnames
            assert f"{name}::b_log__" in trace.varnames


class TestMHKernel(SeededTest):
    def test_normal_model(self):
        data = st.norm(10, 0.5).rvs(1000, random_state=self.get_random_state())

        initial_rng_state = np.random.get_state()
        with pm.Model() as m:
            mu = pm.Normal("mu", 0, 3)
            sigma = pm.HalfNormal("sigma", 1)
            y = pm.Normal("y", mu, sigma, observed=data)
            idata = pm.sample_smc(draws=2000, kernel=pm.smc.MH)
        assert_random_state_equal(initial_rng_state, np.random.get_state())

        post = idata.posterior.stack(sample=("chain", "draw"))
        assert np.abs(post["mu"].mean() - 10) < 0.1
        assert np.abs(post["sigma"].mean() - 0.5) < 0.05

    def test_proposal_dist_shape(self):
        with pm.Model() as m:
            x = pm.Normal("x", 0, 1)
            y = pm.Normal("y", x, 1, observed=0)
            trace = pm.sample_smc(
                draws=10,
                chains=1,
                kernel=pm.smc.MH,
                return_inferencedata=False,
            )
