from pymc3 import *

import theano.tensor as tt
import numpy as np


def numpy_invlogit(x):
    import numpy as np
    return np.exp(x) / (1 + np.exp(x))

npred = 4
n = 4000

effects_a = np.random.normal(size=npred)
predictors = np.random.normal(size=(n, npred))


outcomes = np.random.binomial(
    1, numpy_invlogit(np.sum(effects_a[None, :] * predictors, 1)))


model = Model()

with model:
    effects = Normal('effects', mu=0, tau=2. ** -2, shape=(1, npred))
    p = invlogit(sum(effects * predictors, 1))

    o = Bernoulli('o', p, observed=outcomes)


def run(n=3000):
    if n == "short":
        n = 50
    with model:
        # move the chain to the MAP which should be a good starting point
        start = find_MAP()
        step = NUTS(scaling=start)

        trace = sample(n, step, start)

if __name__ == '__main__':
    run()
