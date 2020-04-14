import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
seed = 1234

np.random.seed(seed)
size = 100
true_mean = 11.0

y = np.array([true_mean])

with pm.Model() as model0:
    sigma = 1.0
    x_coeff = pm.Normal('x', true_mean, sigma=100.0)
    # Define likelihood
    likelihood = pm.Normal('y', mu=x_coeff,
                           sigma=sigma, observed=y+1.0)

with pm.Model() as model1:
    sigma = 1.0
    x_coeff = pm.Normal('x', true_mean, sigma=100.0)
    # Define likelihood
    likelihood = pm.Normal('y', mu=x_coeff,
                           sigma=sigma, observed=y+0.5)

coarse_models = [model0, model1]

with pm.Model() as model:
    sigma = 1.0
    x_coeff = pm.Normal('x', true_mean, sigma=100.0)
    # Define likelihood
    likelihood = pm.Normal('y', mu=x_coeff,
                           sigma=sigma, observed=y)
    # Inference!
    step_temp = pm.MLDA(subsampling_rate=2, coarse_models=coarse_models)
    trace = pm.sample(500, chains=2, tune=False, step=step_temp, random_seed=seed)
    trace2 = pm.sample(500, chains=2, tune=False, random_seed=seed)

    print(pm.stats.summary(trace))
    print(pm.stats.summary(trace2))
    print("Acceptance: " + str(trace.get_sampler_stats('accepted').mean()))

    pm.plots.traceplot(trace)
    pm.plots.traceplot(trace2)
    plt.show()
