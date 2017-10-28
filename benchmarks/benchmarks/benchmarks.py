import time

import numpy as np
import pandas as pd
import pymc3 as pm
import theano


class OverheadSuite(object):
    """
    Just tests how long sampling from a normal distribution takes for various
    samplers
    """
    params = [pm.NUTS, pm.HamiltonianMC, pm.Metropolis, pm.Slice]

    def setup(self, step):
        self.n_steps = 10000
        with pm.Model() as self.model:
            pm.Normal('x', mu=0, sd=1)

    def time_overhead_sample(self, step):
        with self.model:
            pm.sample(self.n_steps, step=step(), random_seed=1)


class ExampleSuite(object):
    """Implements examples to keep up with benchmarking them."""
    timeout = 360.0  # give it a few minutes

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
            pm.sample(2000, njobs=4)

    def time_glm_hierarchical(self):
        data = pd.read_csv(pm.get_data('radon.csv'))
        data['log_radon'] = data['log_radon'].astype(theano.config.floatX)
        county_idx = data.county_code.values

        n_counties = len(data.county.unique())
        with pm.Model():
            mu_a = pm.Normal('mu_a', mu=0., sd=100**2)
            sigma_a = pm.HalfCauchy('sigma_a', 5)
            mu_b = pm.Normal('mu_b', mu=0., sd=100**2)
            sigma_b = pm.HalfCauchy('sigma_b', 5)
            a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=n_counties)
            b = pm.Normal('b', mu=mu_b, sd=sigma_b, shape=n_counties)
            eps = pm.HalfCauchy('eps', 5)
            radon_est = a[county_idx] + b[county_idx] * data.floor.values
            pm.Normal('radon_like', mu=radon_est, sd=eps, observed=data.log_radon)
            pm.sample(draws=2000, njobs=4)


class EffectiveSampleSizeSuite(object):
    """Tests effective sample size per second on models
    """
    timeout = 360.0
    params = (
        [pm.NUTS, pm.Metropolis],  # Slice too slow, don't want to tune HMC
        ['advi', 'jitter+adapt_diag', 'advi+adapt_diag_grad'],
    )
    param_names = ['step', 'init']

    def setup(self, step, init):
        """Initialize model and get start position"""
        np.random.seed(123)
        self.chains = 4
        data = pd.read_csv(pm.get_data('radon.csv'))
        data['log_radon'] = data['log_radon'].astype(theano.config.floatX)
        county_idx = data.county_code.values
        n_counties = len(data.county.unique())
        with pm.Model() as self.model:
            mu_a = pm.Normal('mu_a', mu=0., sd=100**2)
            sigma_a = pm.HalfCauchy('sigma_a', 5)

            mu_b = pm.Normal('mu_b', mu=0., sd=100**2)
            sigma_b = pm.HalfCauchy('sigma_b', 5)

            a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=n_counties)
            b = pm.Normal('b', mu=mu_b, sd=sigma_b, shape=n_counties)
            eps = pm.HalfCauchy('eps', 5)

            radon_est = a[county_idx] + b[county_idx] * data.floor.values

            pm.Normal('radon_like', mu=radon_est, sd=eps, observed=data.log_radon)
            self.start, _ = pm.init_nuts(chains=self.chains, init=init)

    def track_glm_hierarchical_ess(self, step, init):
        with self.model:
            t0 = time.time()
            trace = pm.sample(draws=20000, step=step(), njobs=4, chains=self.chains,
                              start=self.start, random_seed=100)
            tot = time.time() - t0
        ess = pm.effective_n(trace, ('mu_a',))['mu_a']
        return ess / tot

EffectiveSampleSizeSuite.track_glm_hierarchical_ess.unit = 'Effective samples per second'
