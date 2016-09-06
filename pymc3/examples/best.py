"""
Bayesian Estimation Supersedes the T-Test

This model replicates the example used in:
Kruschke, John. (2012) Bayesian estimation supersedes the t test.
Journal of Experimental Psychology: General.

The original pymc2 implementation was written by Andrew Straw
and can be found here:
https://github.com/strawlab/best

Ported to PyMC3 by Thomas Wiecki (c) 2015.
"""

import numpy as np

import pymc3 as pm

drug = (101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100,
        123, 105, 103, 100, 95, 102, 106,
        109, 102, 82, 102, 100, 102, 102, 101, 102, 102, 103,
        103, 97, 97, 103, 101, 97, 104,
        96, 103, 124, 101, 101, 100, 101, 101, 104, 100, 101)
placebo = (99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102,
           102, 100, 105, 88, 101, 100, 104, 100, 100, 100,
           101, 102, 103, 97, 101, 101,
           100, 101, 99, 101, 100, 100,
           101, 100, 99, 101, 100, 102, 99, 100, 99)

y1 = np.array(drug)
y2 = np.array(placebo)
y = np.concatenate((y1, y2))

mu_m = np.mean(y)
mu_p = 0.000001 * 1 / np.std(y) ** 2

sigma_low = np.std(y) / 1000
sigma_high = np.std(y) * 1000

with pm.Model() as model:
    group1_mean = pm.Normal('group1_mean', mu=mu_m, tau=mu_p,
                            testval=y1.mean())
    group2_mean = pm.Normal('group2_mean', mu=mu_m, tau=mu_p,
                            testval=y2.mean())
    group1_std = pm.Uniform('group1_std', lower=sigma_low, upper=sigma_high,
                            testval=y1.std())
    group2_std = pm.Uniform('group2_std', lower=sigma_low, upper=sigma_high,
                            testval=y2.std())
    nu = pm.Exponential('nu_minus_one', 1 / 29.) + 1

    lam1 = group1_std ** -2
    lam2 = group2_std ** -2

    group1 = pm.StudentT(
        'drug', nu=nu, mu=group1_mean, lam=lam1, observed=y1)
    group2 = pm.StudentT(
        'placebo', nu=nu, mu=group2_mean, lam=lam2,
        observed=y2)

    diff_of_means = pm.Deterministic(
        'difference of means', group1_mean -
        group2_mean)
    diff_of_stds = pm.Deterministic(
        'difference of stds',
        group1_std - group2_std)
    effect_size = pm.Deterministic(
        'effect size',
        diff_of_means / pm.sqrt(
            (group1_std **
             2 + group2_std**2) / 2))

    step = pm.NUTS()


def run(n=3000):
    if n == "short":
        n = 500
    with model:
        trace = pm.sample(n, step)

    burn = n / 10

    pm.traceplot(trace[burn:])
    pm.plots.summary(trace[burn:])


if __name__ == '__main__':
    run()
