import pymc3 as pm
import numpy as np
from numpy import random as nr
import numpy.testing as npt
import pytest
import theano.tensor as tt
import theano

from pymc3.distributions.distribution import _draw_value, draw_values
from .helpers import SeededTest


def test_draw_value():
    npt.assert_equal(_draw_value(np.array([5, 6])), [5, 6])
    npt.assert_equal(_draw_value(np.array(5.)), 5)

    npt.assert_equal(_draw_value(tt.constant([5., 6.])), [5, 6])
    assert _draw_value(tt.constant(5)) == 5
    npt.assert_equal(_draw_value(2 * tt.constant([5., 6.])), [10, 12])

    val = theano.shared(np.array([5., 6.]))
    npt.assert_equal(_draw_value(val), [5, 6])
    npt.assert_equal(_draw_value(2 * val), [10, 12])

    a = tt.scalar('a')
    a.tag.test_value = 6
    npt.assert_equal(_draw_value(2 * a, givens=[(a, 1)]), 2)

    assert _draw_value(5) == 5
    assert _draw_value(5.) == 5
    assert isinstance(_draw_value(5.), type(5.))
    assert isinstance(_draw_value(5), type(5))

    with pm.Model():
        mu = 2 * tt.constant(np.array([5., 6.])) + theano.shared(np.array(5))
        a = pm.Normal('a', mu=mu, sigma=5, shape=2)

    val1 = _draw_value(a)
    val2 = _draw_value(a)
    assert np.all(val1 != val2)

    with pytest.raises(ValueError) as err:
        _draw_value([])
    err.match('Unexpected type')


class TestDrawValues:
    def test_empty(self):
        assert draw_values([]) == []

    def test_vals(self):
        npt.assert_equal(draw_values([np.array([5, 6])])[0], [5, 6])
        npt.assert_equal(draw_values([np.array(5.)])[0], 5)

        npt.assert_equal(draw_values([tt.constant([5., 6.])])[0], [5, 6])
        assert draw_values([tt.constant(5)])[0] == 5
        npt.assert_equal(draw_values([2 * tt.constant([5., 6.])])[0], [10, 12])

        val = theano.shared(np.array([5., 6.]))
        npt.assert_equal(draw_values([val])[0], [5, 6])
        npt.assert_equal(draw_values([2 * val])[0], [10, 12])

    def test_simple_model(self):
        with pm.Model():
            mu = 2 * tt.constant(np.array([5., 6.])) + theano.shared(np.array(5))
            a = pm.Normal('a', mu=mu, sigma=5, shape=2)

        val1 = draw_values([a])
        val2 = draw_values([a])
        assert np.all(val1[0] != val2[0])

        point = {'a': np.array([3., 4.])}
        npt.assert_equal(draw_values([a], point=point), [point['a']])

    def test_dep_vars(self):
        with pm.Model():
            mu = 2 * tt.constant(np.array([5., 6.])) + theano.shared(np.array(5))
            sd = pm.HalfNormal('sd', shape=2)
            tau = 1 / sd ** 2
            a = pm.Normal('a', mu=mu, tau=tau, shape=2)

        point = {'a': np.array([1., 2.])}
        npt.assert_equal(draw_values([a], point=point), [point['a']])

        val1 = draw_values([a])[0]
        val2 = draw_values([a], point={'sd': np.array([2., 3.])})[0]
        val3 = draw_values([a], point={'sd_log__': np.array([2., 3.])})[0]
        val4 = draw_values([a], point={'sd_log__': np.array([2., 3.])})[0]

        assert all([np.all(val1 != val2), np.all(val1 != val3),
                    np.all(val1 != val4), np.all(val2 != val3),
                    np.all(val2 != val4), np.all(val3 != val4)])


class TestJointDistributionDrawValues(SeededTest):
    def test_joint_distribution(self):
        with pm.Model() as model:
            a = pm.Normal('a', mu=0, sigma=100)
            b = pm.Normal('b', mu=a, sigma=1e-8)
            c = pm.Normal('c', mu=a, sigma=1e-8)
            d = pm.Deterministic('d', b + c)

        # Expected RVs
        N = 1000
        norm = np.random.randn(3, N)
        eA = norm[0] * 100
        eB = eA + norm[1] * 1e-8
        eC = eA + norm[2] * 1e-8
        eD = eB + eC

        # Drawn RVs
        nr.seed(self.random_seed)
#        A, B, C, D = list(zip(*[draw_values([a, b, c, d]) for i in range(N)]))
        A, B, C, D = draw_values([a, b, c, d], size=N)
        A = np.array(A).flatten()
        B = np.array(B).flatten()
        C = np.array(C).flatten()
        D = np.array(D).flatten()

        # Assert that the drawn samples match the expected values
        assert np.allclose(eA, A)
        assert np.allclose(eB, B)
        assert np.allclose(eC, C)
        assert np.allclose(eD, D)

        # Assert that A, B and C have the expected difference
        assert np.all(np.abs(A - B) < 1e-6)
        assert np.all(np.abs(A - C) < 1e-6)
        assert np.all(np.abs(B - C) < 1e-6)

        # Marginal draws
        mA = np.array([draw_values([a]) for i in range(N)]).flatten()
        mB = np.array([draw_values([b]) for i in range(N)]).flatten()
        mC = np.array([draw_values([c]) for i in range(N)]).flatten()
        # Also test the with model context of draw_values
        with model:
            mD = np.array([draw_values([d]) for i in range(N)]).flatten()

        # Assert that the marginal distributions have different sample values
        assert not np.all(np.abs(B - mB) < 1e-2)
        assert not np.all(np.abs(C - mC) < 1e-2)
        assert not np.all(np.abs(D - mD) < 1e-2)

        # Assert that the marginal distributions do not have high cross
        # correlation
        assert np.abs(np.corrcoef(mA, mB)[0, 1]) < 0.1
        assert np.abs(np.corrcoef(mA, mC)[0, 1]) < 0.1
        assert np.abs(np.corrcoef(mB, mC)[0, 1]) < 0.1
