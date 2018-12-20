import time
import timeit

import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as tt


def glm_hierarchical_model(random_seed=123):
    """Sample glm hierarchical model to use in benchmarks"""
    np.random.seed(random_seed)
    data = pd.read_csv(pm.get_data('radon.csv'))
    data['log_radon'] = data['log_radon'].astype(theano.config.floatX)
    county_idx = data.county_code.values

    n_counties = len(data.county.unique())
    with pm.Model() as model:
        mu_a = pm.Normal('mu_a', mu=0., sd=100**2)
        sigma_a = pm.HalfCauchy('sigma_a', 5)
        mu_b = pm.Normal('mu_b', mu=0., sd=100**2)
        sigma_b = pm.HalfCauchy('sigma_b', 5)
        a = pm.Normal('a', mu=0, sd=1, shape=n_counties)
        b = pm.Normal('b', mu=0, sd=1, shape=n_counties)
        a = mu_a + sigma_a * a
        b = mu_b + sigma_b * b
        eps = pm.HalfCauchy('eps', 5)
        radon_est = a[county_idx] + b[county_idx] * data.floor.values
        pm.Normal('radon_like', mu=radon_est, sd=eps, observed=data.log_radon)
    return model


def mixture_model(random_seed=1234):
    """Sample mixture model to use in benchmarks"""
    np.random.seed(1234)
    size = 1000
    w_true = np.array([0.35, 0.4, 0.25])
    mu_true = np.array([0., 2., 5.])
    sigma = np.array([0.5, 0.5, 1.])
    component = np.random.choice(mu_true.size, size=size, p=w_true)
    x = np.random.normal(mu_true[component], sigma[component], size=size)

    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones_like(w_true))
        mu = pm.Normal('mu', mu=0., sd=10., shape=w_true.shape)
        enforce_order = pm.Potential('enforce_order', tt.switch(mu[0] - mu[1] <= 0, 0., -np.inf) +
                                                      tt.switch(mu[1] - mu[2] <= 0, 0., -np.inf))
        tau = pm.Gamma('tau', alpha=1., beta=1., shape=w_true.shape)
        pm.NormalMixture('x_obs', w=w, mu=mu, tau=tau, observed=x)

    # Initialization can be poorly specified, this is a hack to make it work
    start = {
        'mu': mu_true.copy(),
        'tau_log__': np.log(1. / sigma**2),
        'w_stickbreaking__': np.array([-0.03,  0.44])
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
            pm.Normal('x', mu=0, sd=1)

    def time_overhead_sample(self, step):
        with self.model:
            pm.sample(self.n_steps, step=step(), random_seed=1,
                      progressbar=False, compute_convergence_checks=False)


class ExampleSuite:
    """Implements examples to keep up with benchmarking them."""
    timeout = 360.0  # give it a few minutes
    timer = timeit.default_timer

    def time_drug_evaluation(self):
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

        y = pd.DataFrame({
            'value': np.r_[drug, placebo],
            'group': np.r_[['drug']*len(drug), ['placebo']*len(placebo)]
            })
        y_mean = y.value.mean()
        y_std = y.value.std() * 2

        sigma_low = 1
        sigma_high = 10
        with pm.Model():
            group1_mean = pm.Normal('group1_mean', y_mean, sd=y_std)
            group2_mean = pm.Normal('group2_mean', y_mean, sd=y_std)
            group1_std = pm.Uniform('group1_std', lower=sigma_low, upper=sigma_high)
            group2_std = pm.Uniform('group2_std', lower=sigma_low, upper=sigma_high)
            lambda_1 = group1_std**-2
            lambda_2 = group2_std**-2

            nu = pm.Exponential('ν_minus_one', 1/29.) + 1

            pm.StudentT('drug', nu=nu, mu=group1_mean, lam=lambda_1, observed=drug)
            pm.StudentT('placebo', nu=nu, mu=group2_mean, lam=lambda_2, observed=placebo)
            diff_of_means = pm.Deterministic('difference of means', group1_mean - group2_mean)
            pm.Deterministic('difference of stds', group1_std - group2_std)
            pm.Deterministic(
                'effect size', diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2))
            pm.sample(draws=20000, cores=4, chains=4,
                      progressbar=False, compute_convergence_checks=False)

    def time_glm_hierarchical(self):
        with glm_hierarchical_model():
            pm.sample(draws=20000, cores=4, chains=4,
                      progressbar=False, compute_convergence_checks=False)


class NUTSInitSuite:
    """Tests initializations for NUTS sampler on models
    """
    timeout = 360.0
    params = ('adapt_diag', 'jitter+adapt_diag', 'advi+adapt_diag_grad')
    number = 1
    repeat = 1
    draws = 10000
    chains = 4

    def time_glm_hierarchical_init(self, init):
        """How long does it take to run the initialization."""
        with glm_hierarchical_model():
            pm.init_nuts(init=init, chains=self.chains, progressbar=False)

    def track_glm_hierarchical_ess(self, init):
        with glm_hierarchical_model():
            start, step = pm.init_nuts(init=init, chains=self.chains, progressbar=False, random_seed=123)
            t0 = time.time()
            trace = pm.sample(draws=self.draws, step=step, cores=4, chains=self.chains,
                              start=start, random_seed=100, progressbar=False,
                              compute_convergence_checks=False)
            tot = time.time() - t0
        ess = pm.effective_n(trace, ('mu_a',))['mu_a']
        return ess / tot

    def track_marginal_mixture_model_ess(self, init):
        model, start = mixture_model()
        with model:
            _, step = pm.init_nuts(init=init, chains=self.chains,
                                   progressbar=False, random_seed=123)
            start = [{k: v for k, v in start.items()} for _ in range(self.chains)]
            t0 = time.time()
            trace = pm.sample(draws=self.draws, step=step, cores=4, chains=self.chains,
                              start=start, random_seed=100, progressbar=False,
                              compute_convergence_checks=False)
            tot = time.time() - t0
        ess = pm.effective_n(trace, ('mu',))['mu'].min()  # worst case
        return ess / tot


NUTSInitSuite.track_glm_hierarchical_ess.unit = 'Effective samples per second'
NUTSInitSuite.track_marginal_mixture_model_ess.unit = 'Effective samples per second'


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
            trace = pm.sample(draws=self.draws, step=step, cores=4, chains=4,
                              random_seed=100, progressbar=False,
                              compute_convergence_checks=False)
            tot = time.time() - t0
        ess = pm.effective_n(trace, ('mu_a',))['mu_a']
        return ess / tot


CompareMetropolisNUTSSuite.track_glm_hierarchical_ess.unit = 'Effective samples per second'
