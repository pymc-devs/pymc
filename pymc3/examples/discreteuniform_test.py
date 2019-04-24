import numpy as np
from pymc3 import *


def run_MCMC(lower, upper, obs, draws):
    """Define and run MCMC model with using a DiscreteUniform distribution.
    This may be useful in place of a categorical distribution for a very large support
    (i.e. large n in example below)."""
    obs = np.array(obs)
    with Model() as model2:
        x = DiscreteUniform('x', lower, upper - 1)
        sfs_obs = Poisson('sfs_obs', mu=x, observed=obs)

    with model2:
        step = CategoricalGibbsMetropolis([x])
        trace = sample(draws, tune=0, step=step)  # Will set to CategoricalGibbsMetropolis
    plt.show(forestplot(trace, varnames=['x']))
    traceplot(trace, varnames=['x'])
    print(summary(trace, varnames=['x']))
    return trace


obs = 5000000
draws = 20000
n = 20000000
trace1 = run_MCMC(0, n, obs, draws)
trace2 = run_MCMC(1, n - 1, obs, draws)  # should cause error