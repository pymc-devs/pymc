import pymc3 as pm

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
seed = 12345

np.random.seed(seed)
size = 100
true_mean = 2.0

y = np.random.normal(scale=1.0, size=size) + true_mean


with pm.Model() as model0:
    # Define priors
    sigma = 1.0  # pm.HalfCauchy('sigma', beta=10, testval=1.)
    #intercept = true_intercept  # pm.Normal('Intercept', 0, sigma=20)
    x_coeff = pm.Normal('x', true_mean, sigma=1000.0)

    # Define likelihood
    likelihood = pm.Normal('y', mu=x_coeff + 1.0,
                           sigma=sigma, observed=y)

with pm.Model() as model1:
    # Define priors
    sigma = 1.0  # pm.HalfCauchy('sigma', beta=10, testval=1.)
    # intercept = true_intercept  # pm.Normal('Intercept', 0, sigma=20)
    x_coeff = pm.Normal('x', true_mean, sigma=1000.0)

    # Define likelihood
    likelihood = pm.Normal('y', mu=x_coeff + 0.5,
                           sigma=sigma, observed=y)

coarse_models = [model0, model1]

with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = 1.0  # pm.HalfCauchy('sigma', beta=10, testval=1.)
    # intercept = true_intercept  # pm.Normal('Intercept', 0, sigma=20)
    x_coeff = pm.Normal('x', true_mean, sigma=1000.0)

    # Define likelihood
    likelihood = pm.Normal('y', mu=x_coeff,
                           sigma=sigma, observed=y)

    # Inference!
    # trace = pm.sample(1000, cores=2, **{"ml_models": [1, 2, 3]}) # draw 3000 posterior samples using NUTS sampling
    # step_temp = pm.MLDA(subsampling_rate=2, **{"coarse_models": coarse_models})
    step_temp = pm.MLDA(subsampling_rate=10, coarse_models=coarse_models)
    trace = pm.sample(50, chains=1, cores=1, tune=20, step=step_temp, random_seed=seed)
    trace2 = pm.sample(50, chains=1, cores=1, tune=20, random_seed=seed)

    # import ipdb; ipdb.set_trace()
    # plt.hist(trace.get_values(varname='x'))
    # plt.show()

    print(pm.stats.summary(trace))
    print(pm.stats.summary(trace2))

    pm.plots.traceplot(trace)
    plt.show()