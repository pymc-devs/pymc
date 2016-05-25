from pymc3 import Normal, sample, Model, Bound, traceplot, plots, find_MAP
from pymc3 import *
from pymc3.distributions.timeseries import GARCH11
from scipy import optimize

import theano.tensor as T
from theano import shared
import numpy as np

"""
Example from STAN - slightly altered
// GARCH(1,1)

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

Ported to PyMC3 by Peadar Coyle and Olaf Smits (c) 2016.
"""
r = shared(np.array([28, 8, -3, 7, -1, 1, 18, 12, 15], dtype=np.float32))
sigma1 = shared(np.array(.2, dtype=np.float32))
alpha0 = shared(np.array(.5, dtype=np.float32))

with Model() as garch:
    alpha1 = Normal('alpha1', 0., 1., dtype='float32')
    BoundedNormal = Bound(Normal, lower=0., upper=(1 - alpha1))
    beta1 = BoundedNormal('beta1', 0., sd=1e3, dtype='float32')

    obs = GARCH11('garchy_garch', omega=alpha0, alpha_1=alpha1,
                  beta_1=beta1, initial_vol=sigma1, observed=r,
                  dtype='float32')


def run(n=1000):
    with garch:
        tr = sample(n)
        start = find_MAP(fmin=optimize.fmin_bfgs)
        trace = sample(n, Slice(), start=start)

    traceplot(trace)
    plots.summary(trace)


if __name__ == '__main__':
    run()
