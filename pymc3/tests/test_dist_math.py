import numpy as np
import theano.tensor as tt
import pymc3 as pm

from ..distributions import Discrete
from ..distributions.dist_math import bound_elemwise, bound, factln


def test_bound_elemwise():
    logp = tt.ones((10, 10))
    cond = tt.ones((10, 10))
    assert np.all(bound_elemwise(logp, cond).eval() == logp.eval())

    logp = tt.ones((10, 10))
    cond = tt.zeros((10, 10))
    assert np.all(bound_elemwise(logp, cond).eval() == (-np.inf * logp).eval())

    logp = tt.ones((10, 10))
    cond = True
    assert np.all(bound_elemwise(logp, cond).eval() == logp.eval())

    logp = tt.ones(3)
    cond = np.array([1, 0, 1])
    assert not np.all(bound_elemwise(logp, cond).eval() == 1)
    assert np.prod(bound_elemwise(logp, cond).eval()) == -np.inf

    logp = tt.ones((2, 3))
    cond = np.array([[1, 1, 1], [1, 0, 1]])
    assert not np.all(bound_elemwise(logp, cond).eval() == 1)
    assert np.prod(bound_elemwise(logp, cond).eval()) == -np.inf


class MultinomialA(Discrete):
    def __init__(self, n, p, *args, **kwargs):
        super(MultinomialA, self).__init__(*args, **kwargs)

        self.n = n
        self.p = p

    def logp(self, value):
        n = self.n
        p = self.p

        return bound(factln(n) - factln(value).sum() + (value * tt.log(p)).sum(),
                     value >= 0,
                     0 <= p, p <= 1,
                     tt.isclose(p.sum(), 1))


class MultinomialB(Discrete):
    def __init__(self, n, p, *args, **kwargs):
        super(MultinomialB, self).__init__(*args, **kwargs)

        self.n = n
        self.p = p

    def logp(self, value):
        n = self.n
        p = self.p

        return bound(factln(n) - factln(value).sum() + (value * tt.log(p)).sum(),
                     tt.all(value >= 0),
                     tt.all(0 <= p), tt.all(p <= 1),
                     tt.isclose(p.sum(), 1))


def test_multinomial_bound():

    x = np.array([1, 5])
    n = x.sum()

    with pm.Model() as modelA:
        p_a = pm.Dirichlet('p', np.ones(2))
        x_obs_a = MultinomialA('x', n, p_a, observed=x)

    with pm.Model() as modelB:
        p_b = pm.Dirichlet('p', np.ones(2))
        x_obs_b = MultinomialB('x', n, p_b, observed=x)

    assert np.isclose(modelA.logp({'p_stickbreaking_': [0]}),
                      modelB.logp({'p_stickbreaking_': [0]}))
