import pymc3 as pm
import numpy as np
import numpy.testing as npt
import pytest
import theano.tensor as tt
import theano

from pymc3.distributions.distribution import _draw_value, draw_values


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
        a = pm.Normal('a', mu=mu, sd=5, shape=2)

    val1 = _draw_value(a)
    val2 = _draw_value(a)
    assert np.all(val1 != val2)

    with pytest.raises(ValueError) as err:
        _draw_value([])
    err.match('Unexpected type')


class TestDrawValues(object):
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
            a = pm.Normal('a', mu=mu, sd=5, shape=2)

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

        with pytest.raises(theano.gof.MissingInputError):
            draw_values([a])

        # We need the untransformed vars
        with pytest.raises(theano.gof.MissingInputError):
            draw_values([a], point={'sd': np.array([2., 3.])})

        val1 = draw_values([a], point={'sd_log__': np.array([2., 3.])})[0]
        val2 = draw_values([a], point={'sd_log__': np.array([2., 3.])})[0]
        assert np.all(val1 != val2)
