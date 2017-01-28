from pymc3 import HalfCauchy, Normal, sample, Model, loo
import numpy as np

'''Original Stan model

data {
  int<lower=0> J; // number of schools
  real y[J]; // estimated treatment effects
  real<lower=0> sigma[J]; // s.e. of effect estimates
}
parameters {
  real mu;
  real<lower=0> tau;
  real eta[J];
}
transformed parameters {
  real theta[J];
  for (j in 1:J)
    theta[j] <- mu + tau * eta[j];
}
model {
  eta ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
'''

J = 8
y = np.array([28,  8, -3,  7, -1,  1, 18, 12])
sigma = np.array([15, 10, 16, 11,  9, 11, 10, 18])

with Model() as schools:

    eta = Normal('eta', 0, 1, shape=J)
    mu = Normal('mu', 0, sd=1e6)
    tau = HalfCauchy('tau', 25)

    theta = mu + tau * eta

    obs = Normal('obs', theta, sd=sigma, observed=y)


def run(n=1000):
    if n == "short":
        n = 50
    with schools:
        tr = sample(n)
        loo(tr)

if __name__ == '__main__':
    run()
