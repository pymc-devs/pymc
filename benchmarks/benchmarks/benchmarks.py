import numpy as np
import pandas as pd
import pymc3 as pm


class OverheadSuite(object):
    """
    Just tests how long sampling from a normal distribution takes for various
    samplers
    """
    params = [pm.NUTS, pm.HamiltonianMC, pm.Metropolis, pm.Slice]

    def setup(self, step):
        self.n_steps = 100000
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
        with pm.Model() as model:
            group1_mean = pm.Normal('group1_mean', y_mean, sd=y_std)
            group2_mean = pm.Normal('group2_mean', y_mean, sd=y_std)
            group1_std = pm.Uniform('group1_std', lower=sigma_low, upper=sigma_high)
            group2_std = pm.Uniform('group2_std', lower=sigma_low, upper=sigma_high)
            lambda_1 = group1_std**-2
            lambda_2 = group2_std**-2

            nu = pm.Exponential('ν_minus_one', 1/29.) + 1

            group1 = pm.StudentT('drug', nu=nu, mu=group1_mean, lam=lambda_1,
                                 observed=drug)
            group2 = pm.StudentT('placebo', nu=nu, mu=group2_mean, lam=lambda_2,
                                 observed=placebo)
            diff_of_means = pm.Deterministic('difference of means', group1_mean - group2_mean)
            diff_of_stds = pm.Deterministic('difference of stds', group1_std - group2_std)
            effect_size = pm.Deterministic(
                'effect size', diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2))
            trace = pm.sample(2000, init=None, njobs=2)
