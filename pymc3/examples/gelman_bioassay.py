from numpy import array, ones

import pymc3 as pm

# Samples for each dose level
n = 5 * ones(4, dtype=int)
# Log-dose
dose = array([-0.86, -0.3, -0.05, 0.73])

with pm.Model() as model:

    # Logit-linear model parameters
    alpha = pm.Normal("alpha", 0, sigma=100.0)
    beta = pm.Normal("beta", 0, sigma=1.0)

    # Calculate probabilities of death
    theta = pm.Deterministic("theta", pm.math.invlogit(alpha + beta * dose))

    # Data likelihood
    deaths = pm.Binomial("deaths", n=n, p=theta, observed=[0, 1, 3, 5])


def run(n=1000):
    if n == "short":
        n = 50
    with model:
        pm.sample(n, tune=1000)


if __name__ == "__main__":
    run()
