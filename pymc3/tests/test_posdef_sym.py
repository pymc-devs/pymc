from pymc3.distributions import multivariate as mv
import theano
import numpy as np


def test_posdef_symmetric1():
    data = np.array([[1., 0], [0, 1]], dtype=theano.config.floatX)
    assert mv.posdef(data) == 1


def test_posdef_symmetric2():
    data = np.array([[1., 2], [2, 1]], dtype=theano.config.floatX)
    assert mv.posdef(data) == 0


def test_posdef_symmetric3():
    """ The test return 0 if the matrix has 0 eigenvalue.

    Is this correct?
    """
    data = np.array([[1., 1], [1, 1]], dtype=theano.config.floatX)
    assert mv.posdef(data) == 0


def test_posdef_symmetric4():
    d = np.array([[1,  .99,  1],
                  [.99, 1,  .999],
                  [1,  .999, 1]], theano.config.floatX)

    assert mv.posdef(d) == 0
