import pymc3 as pm

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

size = 50
true_intercept = 2
true_slope = 5
seed = 12345

np.random.seed(seed)

x = np.linspace(0, 1, size)
true_regression_line = true_intercept + true_slope * x
# add noise
y = true_regression_line + np.random.normal(scale=.5, size=size)

data = dict(x=x, y=y)

with pm.Model() as model0:
    # Define priors
    sigma = .5
    intercept = true_intercept
    x_coeff = pm.Normal('x', 0, sigma=20)

    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept + (x_coeff + 0.02) * x,
                           sigma=sigma, observed=y)

with pm.Model() as model1:
    # Define priors
    sigma = .5
    intercept = true_intercept
    x_coeff = pm.Normal('x', 0, sigma=20)

    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept + (x_coeff + 0.01) * x,
                           sigma=sigma, observed=y)

coarse_models = [model0, model1]

with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = .5
    intercept = true_intercept
    x_coeff = pm.Normal('x', 0, sigma=20)

    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept + x_coeff * x,
                           sigma=sigma, observed=y)

    # Inference!
    step_temp = pm.MLDA(subsampling_rate=3, coarse_models=coarse_models)
    trace = pm.sample(50, chains=1, cores=1, tune=20, step=step_temp, random_seed=seed)
    trace2 = pm.sample(50, chains=1, cores=1, tune=20, random_seed=seed)

    print(pm.stats.summary(trace))
    print(pm.stats.summary(trace2))

    pm.plots.traceplot(trace)
    plt.show()




