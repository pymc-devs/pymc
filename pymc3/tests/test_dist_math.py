import numpy as np
import theano.tensor as tt

from ..distributions.dist_math import bound


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