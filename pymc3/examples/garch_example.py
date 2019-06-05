from pymc3 import Normal, sample, Model, Uniform, summary
import theano.tensor as tt
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
"""


def get_garch_model():
    r = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float64)
    sigma1 = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float64)
    alpha0 = np.array([10, 10, 16, 8, 9, 11, 12, 18], dtype=np.float64)
    shape = r.shape

    with Model() as garch:
        alpha1 = Uniform('alpha1', 0., 1., shape=shape)
        beta1 = Uniform('beta1', 0., 1 - alpha1, shape=shape)
        mu = Normal('mu', mu=0., sigma=100., shape=shape)
        theta = tt.sqrt(alpha0 + alpha1 * tt.pow(r - mu, 2) +
                        beta1 * tt.pow(sigma1, 2))
        Normal('obs', mu, sigma=theta, observed=r)
    return garch


def run(n=1000):
    if n == "short":
        n = 50
    with get_garch_model():
        tr = sample(n, tune=1000)
    return tr


if __name__ == '__main__':
    summary(run())
