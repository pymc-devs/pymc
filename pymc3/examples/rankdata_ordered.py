import numpy as np
import theano.tensor as tt

import pymc3 as pm

"""
Using Ordered transformation to model ranking data
inspired by the Stan implementation of Thurstonian model
see http://discourse.mc-stan.org/t/thurstonian-model/1735/5
also see related discussion on PyMC3 discourse:
https://discourse.pymc.io/t/order-statistics-in-pymc3/617
"""


# data
K = 5  # number of items being ranked
J = 100  # number of raters
yreal = np.argsort(np.random.randn(1, K), axis=-1)
y = np.argsort(yreal + np.random.randn(J, K), axis=-1)

# transformed data
y_argsort = np.argsort(y, axis=-1)


with pm.Model() as m:
    mu_hat = pm.Normal("mu_hat", 0, 1, shape=K - 1)
    # set first value to 0 to avoid unidentified model
    mu = tt.concatenate([[0.0], mu_hat])
    # sd = pm.HalfCauchy('sigma', 1.)
    latent = pm.Normal(
        "latent",
        mu=mu[y_argsort],
        sigma=1.0,  # using sd does not work yet
        transform=pm.distributions.transforms.ordered,
        shape=y_argsort.shape,
        testval=np.repeat(np.arange(K)[:, None], J, axis=1).T,
    )
    # There are some problems using Ordered
    # right now, you need to specify testval


def run(n=1500):
    if n == "short":
        n = 50

    with m:
        trace = pm.sample(n)

    pm.traceplot(trace, varnames=["mu_hat"])

    print("Example observed data: ")
    print(y[:30, :].T)
    print("The true ranking is: ")
    print(yreal.flatten())
    print("The Latent mean is: ")
    latentmu = np.hstack(([0], pm.summary(trace, varnames=["mu_hat"])["mean"].values))
    print(np.round(latentmu, 2))
    print("The estimated ranking is: ")
    print(np.argsort(latentmu))


if __name__ == "__main__":
    run()
