from pymc3 import Normal, sample, Model, Bound, traceplot, plots, NUTS
from pymc3.distributions.timeseries import GARCH
import theano.tensor as T
import numpy as np

"""
GARCH(1,1) example
It is interesting to note just how much more compact this is that the original STAN example

The original implementation is in the STAN documentation by Gelman et al and is reproduced below

A good reference studying GARCH models in depth is  http://www.stat-d.si/mz/mz2.1/posedel.pdf

Example from STAN- slightly altered

data {
  int<lower=0> T;
  real r[T];
  real<lower=0> sigma1;
}
parameters {
  real mu;
  real<lower=0> alpha0;
  real<lower=0,upper=1> alpha1;
  real<lower=0, upper=(1-alpha1)> beta1;
}
transformed parameters {
  real<lower=0> sigma[T];
  sigma[1] <- sigma1;
  for (t in 2:T)
    sigma[t] <- sqrt(alpha0
                     + alpha1 * pow(r[t-1] - mu, 2)
                     + beta1 * pow(sigma[t-1], 2));
}
model {
  r ~ normal(mu,sigma);
}
Ported to PyMC3 by Peadar Coyle (c) 2016.
"""
J = 8
r = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma1 = np.array([15, 10, 16, 11, 9, 11, 10, 18])
alpha0 = np.array([10, 10, 16, 8, 9, 11, 12, 18])

with Model() as garch:
    alpha1 = Normal('alpha1', 0, 1, shape=J)
    BoundedNormal = Bound(Normal, upper=(1 - alpha1))
    beta1 = BoundedNormal('beta1', 0, sd=1e6)
    mu = Normal('mu', 0, sd=1e6)

    theta = T.sqrt(alpha0 + alpha1 * T.pow(r - mu, 2) + beta1 * T.pow(sigma1, 2))

    obs = GARCH('garchy_garch', observed=r)

    step = NUTS()


def run(n=1000):
    if n == "short":
        n = 50
    with garch:
        trace = sample(n, step)

    burn = n / 10

    traceplot(trace[burn:])
    plots.summary(trace[burn:])


if __name__ == '__main__':
    run()
