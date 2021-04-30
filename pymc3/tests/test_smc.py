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

import numpy as np
import pytest
import theano.tensor as tt

import pymc3 as pm

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
            trace = pm.sample_smc(500, start=start)


class TestSMCABC(SeededTest):
    def setup_class(self):
        super().setup_class()
        self.data = np.random.normal(loc=0, scale=1, size=1000)

        def normal_sim(a, b):
            return np.random.normal(a, b, 1000)

        with pm.Model() as self.SMABC_test:
            a = pm.Normal("a", mu=0, sigma=1)
            b = pm.HalfNormal("b", sigma=1)
            s = pm.Simulator(
                "s", normal_sim, params=(a, b), sum_stat="sort", epsilon=1, observed=self.data
            )
            self.s = s

        def quantiles(x):
            return np.quantile(x, [0.25, 0.5, 0.75])

        def abs_diff(eps, obs_data, sim_data):
            return np.mean(np.abs((obs_data - sim_data) / eps))

        with pm.Model() as self.SMABC_test2:
            a = pm.Normal("a", mu=0, sigma=1)
            b = pm.HalfNormal("b", sigma=1)
            s = pm.Simulator(
                "s",
                normal_sim,
                params=(a, b),
                distance=abs_diff,
                sum_stat=quantiles,
                epsilon=1,
                observed=self.data,
            )

        with pm.Model() as self.SMABC_potential:
            a = pm.Normal("a", mu=0, sigma=1)
            b = pm.HalfNormal("b", sigma=1)
            c = pm.Potential("c", pm.math.switch(a > 0, 0, -np.inf))
            s = pm.Simulator(
                "s", normal_sim, params=(a, b), sum_stat="sort", epsilon=1, observed=self.data
            )

        with pm.Model() as self.SMABC_log_pseudolikelihood:
            a = pm.Normal("a", mu=0, sigma=1)
            b = pm.HalfNormal("b", sigma=10)
            s = pm.Simulator(
                "s", normal_sim, params=(a, b), sum_stat="mean", epsilon=0.2, observed=self.data
            )

    def test_one_gaussian(self):
        with self.SMABC_test:
            trace = pm.sample_smc(draws=1000, kernel="ABC")

        np.testing.assert_almost_equal(self.data.mean(), trace["a"].mean(), decimal=2)
        np.testing.assert_almost_equal(self.data.std(), trace["b"].mean(), decimal=1)

    def test_sim_data_ppc(self):
        with self.SMABC_test:
            trace, sim_data = pm.sample_smc(draws=1000, kernel="ABC", chains=2, save_sim_data=True)
            pr_p = pm.sample_prior_predictive(1000)
            po_p = pm.sample_posterior_predictive(trace, 1000)

        assert sim_data["s"].shape == (2, 1000, 1000)
        np.testing.assert_almost_equal(self.data.mean(), sim_data["s"].mean(), decimal=2)
        np.testing.assert_almost_equal(self.data.std(), sim_data["s"].std(), decimal=1)
        assert pr_p["s"].shape == (1000, 1000)
        np.testing.assert_almost_equal(0, pr_p["s"].mean(), decimal=1)
        np.testing.assert_almost_equal(1.4, pr_p["s"].std(), decimal=1)
        assert po_p["s"].shape == (1000, 1000)
        np.testing.assert_almost_equal(0, po_p["s"].mean(), decimal=2)
        np.testing.assert_almost_equal(1, po_p["s"].std(), decimal=1)

    def test_custom_dist_sum(self):
        with self.SMABC_test2:
            trace = pm.sample_smc(draws=1000, kernel="ABC")

    def test_potential(self):
        with self.SMABC_potential:
            trace = pm.sample_smc(draws=1000, kernel="ABC")
            assert np.all(trace["a"] >= 0)

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

    def test_repr_latex(self):
        expected = "$\\text{s} \\sim  \\text{Simulator}(\\text{normal_sim}(a, b), \\text{gaussian}, \\text{sort})$"
        assert expected == self.s._repr_latex_()
        assert self.s._repr_latex_() == self.s.__latex__()
        assert self.SMABC_test.model._repr_latex_() == self.SMABC_test.model.__latex__()

    def test_name_is_string_type(self):
        with self.SMABC_potential:
            assert not self.SMABC_potential.name
            trace = pm.sample_smc(draws=10, kernel="ABC")
            assert isinstance(trace._straces[0].name, str)

    def test_named_models_are_unsupported(self):
        def normal_sim(a, b):
            return np.random.normal(a, b, 1000)

        with pm.Model(name="NamedModel"):
            a = pm.Normal("a", mu=0, sigma=1)
            b = pm.HalfNormal("b", sigma=1)
            c = pm.Potential("c", pm.math.switch(a > 0, 0, -np.inf))
            s = pm.Simulator(
                "s", normal_sim, params=(a, b), sum_stat="sort", epsilon=1, observed=self.data
            )
            with pytest.raises(NotImplementedError, match="named models"):
                pm.sample_smc(draws=10, kernel="ABC")

    def test_log_pseudolikelihood(self):
        with self.SMABC_log_pseudolikelihood:
            trace = pm.sample_smc(draws=1000, chains=2, kernel="ABC")
            assert trace.report.log_pseudolikelihood[0].var() < 1
            assert trace.report.log_pseudolikelihood[1].var() < 1
