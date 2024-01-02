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
import time
import timeit

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt

import pymc as pm


def glm_hierarchical_model(random_seed=123):
    """Sample glm hierarchical model to use in benchmarks"""
    np.random.seed(random_seed)
    data = pd.read_csv(pm.get_data("radon.csv"))
    data["log_radon"] = data["log_radon"].astype(pytensor.config.floatX)
    county_idx = data.county_code.values

    n_counties = len(data.county.unique())
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=0.0, sigma=100**2)
        sigma_a = pm.HalfCauchy("sigma_a", 5)
        mu_b = pm.Normal("mu_b", mu=0.0, sigma=100**2)
        sigma_b = pm.HalfCauchy("sigma_b", 5)
        a = pm.Normal("a", mu=0, sigma=1, shape=n_counties)
        b = pm.Normal("b", mu=0, sigma=1, shape=n_counties)
        a = mu_a + sigma_a * a
        b = mu_b + sigma_b * b
        eps = pm.HalfCauchy("eps", 5)
        radon_est = a[county_idx] + b[county_idx] * data.floor.values
        pm.Normal("radon_like", mu=radon_est, sigma=eps, observed=data.log_radon)
    return model


def mixture_model(random_seed=1234):
    """Sample mixture model to use in benchmarks"""
    np.random.seed(1234)
    size = 1000
    w_true = np.array([0.35, 0.4, 0.25])
    mu_true = np.array([0.0, 2.0, 5.0])
    sigma = np.array([0.5, 0.5, 1.0])
    component = np.random.choice(mu_true.size, size=size, p=w_true)
    x = np.random.normal(mu_true[component], sigma[component], size=size)

    with pm.Model() as model:
        w = pm.Dirichlet("w", a=np.ones_like(w_true))
        mu = pm.Normal("mu", mu=0.0, sigma=10.0, shape=w_true.shape)
        enforce_order = pm.Potential(
            "enforce_order",
            pt.switch(mu[0] - mu[1] <= 0, 0.0, -np.inf)
            + pt.switch(mu[1] - mu[2] <= 0, 0.0, -np.inf),
        )
        tau = pm.Gamma("tau", alpha=1.0, beta=1.0, shape=w_true.shape)
        pm.NormalMixture("x_obs", w=w, mu=mu, tau=tau, observed=x)

    # Initialization can be poorly specified, this is a hack to make it work
    start = {
        "mu": mu_true.copy(),
        "tau_log__": np.log(1.0 / sigma**2),
        "w_stickbreaking__": np.array([-0.03, 0.44]),
    }
    return model, start


class OverheadSuite:
    """
    Just tests how long sampling from a normal distribution takes for various
    samplers
    """

    params = [pm.NUTS, pm.HamiltonianMC, pm.Metropolis, pm.Slice]
    timer = timeit.default_timer

    def setup(self, step):
        self.n_steps = 10000
        with pm.Model() as self.model:
            pm.Normal("x", mu=0, sigma=1)

    def time_overhead_sample(self, step):
        with self.model:
            pm.sample(
                self.n_steps,
                step=step(),
                random_seed=1,
                progressbar=False,
                compute_convergence_checks=False,
            )


class ExampleSuite:
    """Implements examples to keep up with benchmarking them."""

    timeout = 360.0  # give it a few minutes
    timer = timeit.default_timer

    def time_drug_evaluation(self):
        # fmt: off
        drug = np.array([101, 100, 102, 104, 102, 97, 105, 105, 98, 101,
                         100, 123, 105, 103, 100, 95, 102, 106, 109, 102, 82,
                         102, 100, 102, 102, 101, 102, 102, 103, 103, 97, 97,
                         103, 101, 97, 104, 96, 103, 124, 101, 101, 100, 101,
                         101, 104, 100, 101])
        placebo = np.array([99, 101, 100, 101, 102, 100, 97, 101, 104, 101,
                            102, 102, 100, 105, 88, 101, 100, 104, 100, 100,
                            100, 101, 102, 103, 97, 101, 101, 100, 101, 99,
                            101, 100, 100, 101, 100, 99, 101, 100, 102, 99,
                            100, 99])
        # fmt: on

        y = pd.DataFrame(
            {
                "value": np.r_[drug, placebo],
                "group": np.r_[["drug"] * len(drug), ["placebo"] * len(placebo)],
            }
        )
        y_mean = y.value.mean()
        y_std = y.value.std() * 2

        sigma_low = 1
        sigma_high = 10
        with pm.Model():
            group1_mean = pm.Normal("group1_mean", y_mean, sigma=y_std)
            group2_mean = pm.Normal("group2_mean", y_mean, sigma=y_std)
            group1_std = pm.Uniform("group1_std", lower=sigma_low, upper=sigma_high)
            group2_std = pm.Uniform("group2_std", lower=sigma_low, upper=sigma_high)
            lambda_1 = group1_std**-2
            lambda_2 = group2_std**-2

            nu = pm.Exponential("ν_minus_one", 1 / 29.0) + 1

            pm.StudentT("drug", nu=nu, mu=group1_mean, lam=lambda_1, observed=drug)
            pm.StudentT("placebo", nu=nu, mu=group2_mean, lam=lambda_2, observed=placebo)
            diff_of_means = pm.Deterministic("difference of means", group1_mean - group2_mean)
            pm.Deterministic("difference of stds", group1_std - group2_std)
            pm.Deterministic(
                "effect size", diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2)
            )
            pm.sample(
                draws=20000, cores=4, chains=4, progressbar=False, compute_convergence_checks=False
            )

    def time_glm_hierarchical(self):
        with glm_hierarchical_model():
            pm.sample(
                draws=20000, cores=4, chains=4, progressbar=False, compute_convergence_checks=False
            )


class NUTSInitSuite:
    """Tests initializations for NUTS sampler on models"""

    timeout = 360.0
    params = ("adapt_diag", "jitter+adapt_diag", "jitter+adapt_full", "adapt_full")
    number = 1
    repeat = 1
    draws = 10000
    chains = 4

    def time_glm_hierarchical_init(self, init):
        """How long does it take to run the initialization."""
        with glm_hierarchical_model():
            pm.init_nuts(
                init=init,
                chains=self.chains,
                progressbar=False,
                random_seed=np.arange(self.chains),
            )

    def track_glm_hierarchical_ess(self, init):
        with glm_hierarchical_model():
            start, step = pm.init_nuts(
                init=init, chains=self.chains, progressbar=False, random_seed=np.arange(self.chains)
            )
            t0 = time.time()
            idata = pm.sample(
                draws=self.draws,
                step=step,
                cores=4,
                chains=self.chains,
                start=start,
                seeds=np.arange(self.chains),
                progressbar=False,
                compute_convergence_checks=False,
            )
            tot = time.time() - t0
        ess = float(az.ess(idata, var_names=["mu_a"])["mu_a"].values)
        return ess / tot

    def track_marginal_mixture_model_ess(self, init):
        model, start = mixture_model()
        with model:
            _, step = pm.init_nuts(
                init=init, chains=self.chains, progressbar=False, random_seed=np.arange(self.chains)
            )
            start = [{k: v for k, v in start.items()} for _ in range(self.chains)]
            t0 = time.time()
            idata = pm.sample(
                draws=self.draws,
                step=step,
                cores=4,
                chains=self.chains,
                start=start,
                seeds=np.arange(self.chains),
                progressbar=False,
                compute_convergence_checks=False,
            )
            tot = time.time() - t0
        ess = az.ess(idata, var_names=["mu"])["mu"].values.min()  # worst case
        return ess / tot


NUTSInitSuite.track_glm_hierarchical_ess.unit = "Effective samples per second"
NUTSInitSuite.track_marginal_mixture_model_ess.unit = "Effective samples per second"


class CompareMetropolisNUTSSuite:
    timeout = 360.0
    # None will be the "sensible default", and include initialization, but should be fastest
    params = (None, pm.NUTS, pm.Metropolis)
    number = 1
    repeat = 1
    draws = 20000

    def track_glm_hierarchical_ess(self, step):
        with glm_hierarchical_model():
            if step is not None:
                step = step()
            t0 = time.time()
            idata = pm.sample(
                draws=self.draws,
                step=step,
                cores=4,
                chains=4,
                random_seed=100,
                progressbar=False,
                compute_convergence_checks=False,
            )
            tot = time.time() - t0
        ess = float(az.ess(idata, var_names=["mu_a"])["mu_a"].values)
        return ess / tot


CompareMetropolisNUTSSuite.track_glm_hierarchical_ess.unit = "Effective samples per second"


class DifferentialEquationSuite:
    """Implements ode examples to keep up with benchmarking them."""

    timeout = 600
    timer = timeit.default_timer

    def track_1var_2par_ode_ess(self):
        def freefall(y, t, p):
            return 2.0 * p[1] - p[0] * y[0]

        # Times for observation
        times = np.arange(0, 10, 0.5)
        y = np.array(
            [
                -2.01,
                9.49,
                15.58,
                16.57,
                27.58,
                32.26,
                35.13,
                38.07,
                37.36,
                38.83,
                44.86,
                43.58,
                44.59,
                42.75,
                46.9,
                49.32,
                44.06,
                49.86,
                46.48,
                48.18,
            ]
        ).reshape(-1, 1)

        ode_model = pm.ode.DifferentialEquation(
            func=freefall, times=times, n_states=1, n_theta=2, t0=0
        )
        with pm.Model() as model:
            # Specify prior distributions for some of our model parameters
            sigma = pm.HalfCauchy("sigma", 1)
            gamma = pm.LogNormal("gamma", 0, 1)
            # If we know one of the parameter values, we can simply pass the value.
            ode_solution = ode_model(y0=[0], theta=[gamma, 9.8])
            # The ode_solution has a shape of (n_times, n_states)
            Y = pm.Normal("Y", mu=ode_solution, sigma=sigma, observed=y)

            t0 = time.time()
            idata = pm.sample(500, tune=1000, chains=2, cores=2, random_seed=0)
            tot = time.time() - t0
        ess = az.ess(idata)
        return np.mean([ess.sigma, ess.gamma]) / tot


DifferentialEquationSuite.track_1var_2par_ode_ess.unit = "Effective samples per second"
