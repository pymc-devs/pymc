from pymc3 import *
import numpy as np

# import pydevd
# pydevd.set_pm_excepthook()


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


start = model.test_point

from theano import ProfileMode


def run(n=1):
    if n == "short":
        n = 50
    for mode in [ProfileMode(linker='py'),
                 ProfileMode(linker='c|py')]:

        print(mode)
        logp = model.logpt
        f = model.fn([logp, gradient(logp)], mode)
        print(f(start))

        # mode.print_summary()

if __name__ == '__main__':
    run()
