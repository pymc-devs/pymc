import pymc3 as pm
from theano import scan, shared

import numpy as np
"""
ARMA example
It is interesting to note just how much more compact this is than the original STAN example

The original implementation is in the STAN documentation by Gelman et al and is reproduced below


Example from STAN- slightly altered

data {
  int<lower=1> T;
  real y[T];
}
parameters {
    // assume err[0] == 0
}
nu[t] <- mu + phi * y[t-1] + theta * err[t-1];
  err[t] <- y[t] - nu[t];
}
mu ~ normal(0,10);
phi ~ normal(0,2);
theta ~ normal(0,2);
  real mu;
  real phi;
  real theta;
  real<lower=0> sigma;
} model {
  vector[T] nu;
  vector[T] err;
  nu[1] <- mu + phi * mu;
  err[1] <- y[1] - nu[1];
  for (t in 2:T) {
    // num observations
    // observed outputs
    // mean coeff
    // autoregression coeff
    // moving avg coeff
    // noise scale
    // prediction for time t
    // error for time t
sigma ~ cauchy(0,5);
err ~ normal(0,sigma);
// priors
// likelihood
Ported to PyMC3 by Peadar Coyle and Chris Fonnesbeck (c) 2016.
"""


def build_model():
    y = shared(np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32))
    with pm.Model() as arma_model:
        sigma = pm.HalfNormal('sigma', 5.)
        theta = pm.Normal('theta', 0., sigma=1.)
        phi = pm.Normal('phi', 0., sigma=2.)
        mu = pm.Normal('mu', 0., sigma=10.)

        err0 = y[0] - (mu + phi * mu)

        def calc_next(last_y, this_y, err, mu, phi, theta):
            nu_t = mu + phi * last_y + theta * err
            return this_y - nu_t

        err, _ = scan(fn=calc_next,
                      sequences=dict(input=y, taps=[-1, 0]),
                      outputs_info=[err0],
                      non_sequences=[mu, phi, theta])

        pm.Potential('like', pm.Normal.dist(0, sigma=sigma).logp(err))
    return arma_model


def run(n_samples=1000):
    model = build_model()
    with model:
        trace = pm.sample(draws=n_samples,
                          tune=1000,
                          target_accept=.99)

    pm.plots.traceplot(trace)
    pm.plots.forestplot(trace)


if __name__ == '__main__':
    run()
