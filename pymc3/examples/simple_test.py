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
# y = a + b*x
true_regression_line = true_intercept + true_slope * x
# add noise
y = true_regression_line + np.random.normal(scale=.5, size=size)

data = dict(x=x, y=y)

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, xlabel='x', ylabel='y', title='Generated data and underlying model')
# ax.plot(x, y, 'x', label='sampled data')
# ax.plot(x, true_regression_line, label='true regression line', lw=2.)
# plt.legend(loc=0);


with pm.Model() as model0:
    # Define priors
    sigma = .5  # pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = true_intercept  # pm.Normal('Intercept', 0, sigma=20)
    x_coeff = pm.Normal('x', 0, sigma=20)

    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept + (x_coeff + 0.02) * x,
                           sigma=sigma, observed=y)

with pm.Model() as model1:
    # Define priors
    sigma = .5  # pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = true_intercept  # pm.Normal('Intercept', 0, sigma=20)
    x_coeff = pm.Normal('x', 0, sigma=20)

    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept + (x_coeff + 0.01) * x,
                           sigma=sigma, observed=y)

coarse_models = [model0, model1]

with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = .5  # pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = true_intercept  # pm.Normal('Intercept', 0, sigma=20)
    x_coeff = pm.Normal('x', 0, sigma=20)

    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept + x_coeff * x,
                           sigma=sigma, observed=y)

    # Inference!
    # trace = pm.sample(1000, cores=2, **{"ml_models": [1, 2, 3]})
    step_temp = pm.MLDA(subsampling_rate=3, coarse_models=coarse_models)
    trace = pm.sample(50, chains=1, cores=1, tune=20, step=step_temp, random_seed=seed)
    trace2 = pm.sample(50, chains=1, cores=1, tune=20, random_seed=seed)

    #import ipdb; ipdb.set_trace()
    #plt.hist(trace.get_values(varname='x'))
    #plt.show()
    
    print(pm.stats.summary(trace))
    print(pm.stats.summary(trace2))

    pm.plots.traceplot(trace)
    plt.show()




