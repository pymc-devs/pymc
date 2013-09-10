from pymc import *

import theano.tensor as t
import numpy as np


def invlogit(x):
    import numpy as np
    return np.exp(x) / (1 + np.exp(x))

npred = 4
n = 4000

effects_a = np.random.normal(size=npred)
predictors = np.random.normal(size=(n, npred))


outcomes = np.random.binomial(
    1, invlogit(np.sum(effects_a[None, :] * predictors, 1)))


def tinvlogit(x):
    import theano.tensor as t
    return t.exp(x) / (1 + t.exp(x))

model = Model()

with model:
    effects = Normal('effects', mu=0, tau=2. ** -2, shape=(1, npred))
    p = tinvlogit(sum(effects * predictors, 1))

    o = Bernoulli('o', p, observed=outcomes)

def run():

    with model:
        # move the chain to the MAP which should be a good starting point
        start = find_MAP()
        step = NUTS(scaling=start)

        trace = sample(3e2, step, start)

if __name__ == '__main__':

    run()
