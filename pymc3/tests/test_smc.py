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

import pymc3 as pm
import numpy as np
import theano.tensor as tt
from .helpers import SeededTest


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
                -0.5 * n * tt.log(2 * np.pi)
                - 0.5 * tt.log(dsigma)
                - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
            )
            log_like2 = (
                -0.5 * n * tt.log(2 * np.pi)
                - 0.5 * tt.log(dsigma)
                - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
            )
            return tt.log(w1 * tt.exp(log_like1) + w2 * tt.exp(log_like2))

        with pm.Model() as self.SMC_test:
            X = pm.Uniform("X", lower=-2, upper=2.0, shape=n)
            llk = pm.Potential("muh", two_gaussians(X))

        self.muref = mu1

    def test_sample(self):
        with self.SMC_test:
            mtrace = pm.sample_smc(draws=self.samples)

        x = mtrace["X"]
        mu1d = np.abs(x).mean(axis=0)
        np.testing.assert_allclose(self.muref, mu1d, rtol=0.0, atol=0.03)

    def test_discrete_continuous(self):
        with pm.Model() as model:
            a = pm.Poisson("a", 5)
            b = pm.HalfNormal("b", 10)
            y = pm.Normal("y", a, b, observed=[1, 2, 3, 4])
            trace = pm.sample_smc()

    def test_ml(self):
        data = np.repeat([1, 0], [50, 50])
        marginals = []
        a_prior_0, b_prior_0 = 1.0, 1.0
        a_prior_1, b_prior_1 = 20.0, 20.0

        for alpha, beta in ((a_prior_0, b_prior_0), (a_prior_1, b_prior_1)):
            with pm.Model() as model:
                a = pm.Beta("a", alpha, beta)
                y = pm.Bernoulli("y", a, observed=data)
                trace = pm.sample_smc(2000)
                marginals.append(model.marginal_log_likelihood)
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
            trace = pm.sample_smc(500, start=start)


class TestSMCABC(SeededTest):
    def setup_class(self):
        super().setup_class()
        self.data = np.random.normal(loc=0, scale=1, size=1000)

        def normal_sim(a, b):
            return np.random.normal(a, b, 1000)

        with pm.Model() as self.SMABC_test:
            a = pm.Normal("a", mu=0, sd=5)
            b = pm.HalfNormal("b", sd=2)
            s = pm.Simulator("s", normal_sim, params=(a, b), observed=self.data)

    def test_one_gaussian(self):
        with self.SMABC_test:
            trace = pm.sample_smc(draws=1000, kernel="ABC", sum_stat="sorted", epsilon=1)

        np.testing.assert_almost_equal(self.data.mean(), trace["a"].mean(), decimal=2)
        np.testing.assert_almost_equal(self.data.std(), trace["b"].mean(), decimal=1)
