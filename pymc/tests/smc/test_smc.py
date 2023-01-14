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
import warnings

import numpy as np
import pytensor.tensor as at
import pytest
import scipy.stats as st

from arviz.data.inference_data import InferenceData

import pymc as pm

from pymc.backends.base import MultiTrace
from pymc.pytensorf import floatX
from pymc.smc.kernels import IMH, systematic_resampling
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

    def test_unobserved_bernoulli(self):
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

    def test_unobserved_categorical(self):
        with pm.Model() as m:
            mu = pm.Categorical("mu", p=[0.1, 0.3, 0.6], size=2)
            pm.Normal("like", mu=mu, sigma=0.1, observed=[1, 2])

            trace = pm.sample_smc(chains=1, return_inferencedata=False)

        assert np.all(np.median(trace["mu"], axis=0) == [1, 2])

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
            # log_marginal_likelihood is found in the last value of each chain
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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                warnings.filterwarnings("ignore", "More chains .* than draws .*", UserWarning)
                idata = pm.sample_smc(chains=chains, draws=draws)
                mt = pm.sample_smc(chains=chains, draws=draws, return_inferencedata=False)

        assert isinstance(idata, InferenceData)
        assert "sample_stats" in idata
        assert idata.posterior.dims["chain"] == chains
        assert idata.posterior.dims["draw"] == draws

        assert isinstance(mt, MultiTrace)
        assert mt.nchains == chains
        assert mt["x"].size == chains * draws

    def test_convergence_checks(self, caplog):
        with caplog.at_level(logging.INFO):
            with self.fast_model:
                pm.sample_smc(draws=99)
        assert "The number of samples is too small" in caplog.text

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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                trace = pm.sample_smc(
                    draws=10,
                    chains=1,
                    kernel=pm.smc.MH,
                    return_inferencedata=False,
                )


def test_systematic():
    rng = np.random.default_rng(seed=34)
    weights = [0.33, 0.33, 0.33]
    np.testing.assert_array_equal(systematic_resampling(weights, rng), [0, 1, 2])
    weights = [0.99, 0.01]
    np.testing.assert_array_equal(systematic_resampling(weights, rng), [0, 0])
