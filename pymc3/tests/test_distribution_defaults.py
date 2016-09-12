from __future__ import division

from ..model import Model
from ..distributions import DiscreteUniform, Continuous

import numpy as np
from nose.tools import raises


class DistTest(Continuous):

    def __init__(self, a, b, *args, **kwargs):
        super(DistTest, self).__init__(*args, **kwargs)
        self.a = a
        self.b = b

    def logp(self, v):
        return 0


@raises(AttributeError)
def test_default_nan_fail():
    with Model():
        DistTest('x', np.nan, 2, defaults=['a'])


@raises(AttributeError)
def test_default_empty_fail():
    with Model():
        DistTest('x', 1, 2, defaults=[])


def test_default_testval():
    with Model():
        x = DistTest('x', 1, 2, testval=5, defaults=[])
        assert x.tag.test_value == 5


def test_default_testval_nan():
    with Model():
        x = DistTest('x', 1, 2, testval=np.nan, defaults=['a'])
        np.testing.assert_almost_equal(x.tag.test_value, np.nan)


def test_default_a():
    with Model():
        x = DistTest('x', 1, 2, defaults=['a'])
        assert x.tag.test_value == 1


def test_default_b():
    with Model():
        x = DistTest('x', np.nan, 2, defaults=['a', 'b'])
        assert x.tag.test_value == 2


def test_default_c():
    with Model():
        y = DistTest('y', 7, 8, testval=94)
        x = DistTest('x', y, 2, defaults=['a', 'b'])
        assert x.tag.test_value == 94


def test_default_discrete_uniform():
    with Model():
        x = DiscreteUniform('x', lower=1, upper=2)
        assert x.init_value == 1
