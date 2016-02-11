from pymc3 import Normal, sample, Model, traceplot, plots, NUTS, Potential, Cauchy, find_MAP, Slice
from theano import scan, shared
from scipy import optimize
import theano.tensor as T


import numpy as np
"""
ARMA example
It is interesting to note just how much more compact this is that the original STAN example

The original implementation is in the STAN documentation by Gelman et al and is reproduced below


Example from STAN- slightly altered

data {
  int<lower=1> T;
  real y[T];
}
parameters {
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
    // assume err[0] == 0
}
nu[t] <- mu + phi * y[t-1] + theta * err[t-1];
  err[t] <- y[t] - nu[t];
}
mu ~ normal(0,10);
phi ~ normal(0,2);
theta ~ normal(0,2);
sigma ~ cauchy(0,5);
err ~ normal(0,sigma);
// priors
// likelihood
Ported to PyMC3 by Peadar Coyle (c) 2016.
"""

t = np.array([1, 2, 4,5,6,8, 19, 18, 12])

y = shared(np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32))


with Model() as arma:

    sigma = Cauchy('sigma', 0, 5)
    theta = Normal('theta', 0, sd=2)
    phi = Normal('phi', 0, sd=2)
    mu = Normal('mu', 0, sd=10)

    nu0 = mu + phi*mu
    err0 = y[0] - nu0

    nu = T.vector('nu')
    err = T.vector('err')


    def calc_next(y, err, mu, phi, theta):
        nu_t = mu + phi*y + theta*err

        err_t = y - nu_t
        return err_t

    err, _ = scan(fn=calc_next,
                          sequences=dict(input=y, taps=[-1]),
                          outputs_info=[dict(initial=err0, taps=[-1])],
                          non_sequences=[mu, phi, theta])
    import ipdb; ipdb.set_trace()
    like = Potential('like', Normal.dist(0, sd=sigma).logp(err))

    step = NUTS()


def run(n=1000):
    if n == "short":
        n = 50
    with arma:

        start = find_MAP(fmin=optimize.fmin_powell)

        trace = sample(n, Slice(), start=start)

    burn = n/10

    traceplot(trace[burn:])
    plots.summary(trace[burn:])


if __name__ == '__main__':
    run()