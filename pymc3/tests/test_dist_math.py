import numpy as np
import theano.tensor as tt
import pymc3 as pm

from ..distributions import Discrete
from ..distributions.dist_math import bound, factln, alltrue_elemwise, alltrue_scalar


def test_bound():
    logp = tt.ones((10, 10))
    cond = tt.ones((10, 10))
    assert np.all(bound(logp, cond).eval() == logp.eval())

    logp = tt.ones((10, 10))
    cond = tt.zeros((10, 10))
    assert np.all(bound(logp, cond).eval() == (-np.inf * logp).eval())

    logp = tt.ones((10, 10))
    cond = True
    assert np.all(bound(logp, cond).eval() == logp.eval())

    logp = tt.ones(3)
    cond = np.array([1, 0, 1])
    assert not np.all(bound(logp, cond).eval() == 1)
    assert np.prod(bound(logp, cond).eval()) == -np.inf

    logp = tt.ones((2, 3))
    cond = np.array([[1, 1, 1], [1, 0, 1]])
    assert not np.all(bound(logp, cond).eval() == 1)
    assert np.prod(bound(logp, cond).eval()) == -np.inf

def test_alltrue_scalar():
    assert alltrue_scalar([]).eval()
    assert alltrue_scalar([True]).eval()
    assert alltrue_scalar([tt.ones(10)]).eval()
    assert alltrue_scalar([tt.ones(10),
                    5 * tt.ones(101)]).eval()
    assert alltrue_scalar([np.ones(10),
                    5 * tt.ones(101)]).eval()
    assert alltrue_scalar([np.ones(10),
                    True,
                    5 * tt.ones(101)]).eval()
    assert alltrue_scalar([np.array([1, 2, 3]),
                    True,
                    5 * tt.ones(101)]).eval()

    assert not alltrue_scalar([False]).eval()
    assert not alltrue_scalar([tt.zeros(10)]).eval()
    assert not alltrue_scalar([True,
                        False]).eval()
    assert not alltrue_scalar([np.array([0, -1]),
                        tt.ones(60)]).eval()
    assert not alltrue_scalar([np.ones(10),
                        False,
                        5 * tt.ones(101)]).eval()

def test_alltrue_shape():
    vals = [True, tt.ones(10), tt.zeros(5)]

    assert alltrue_scalar(vals).eval().shape == ()

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
                     tt.isclose(p.sum(), 1),
                     broadcast_conditions=False
        )


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
                     tt.isclose(p.sum(), 1),
                     broadcast_conditions=False
        )


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
