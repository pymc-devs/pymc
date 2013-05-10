from ..stats import *
from numpy.random import random, normal, seed
from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal

seed(111)
normal_sample = normal(0,1,1000000)

def test_autocorr():
    """Test autocorrelation and autocovariance functions"""

    assert_almost_equal(autocorr(normal_sample), 0, 2)

    y = [(normal_sample[i-1] + normal_sample[i])/2 for i in range(1,len(normal_sample))]
    assert_almost_equal(autocorr(y), 0.5, 2)

def test_hpd():
    """Test HPD calculation"""

    interval = hpd(normal_sample)

    assert_array_almost_equal(interval, [-1.96, 1.96], 2)

def test_make_indices():
    """Test make_indices function"""

    from ..stats import make_indices

    ind = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    assert_equal(ind, make_indices((2,3)))

def test_mc_error():
    """Test batch standard deviation function"""

    x = random(100000)

    assert(mc_error(x) < 0.0025)

def test_quantiles():
    """Test quantiles function"""

    q = quantiles(normal_sample)

    assert_array_almost_equal(q.values(), [-1.96,-0.67,0,0.67,1.96], 2)