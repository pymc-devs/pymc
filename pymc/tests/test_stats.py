import pymc as pm
from pymc import stats
import numpy as np
from numpy.random import random, normal, seed
from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal
import warnings
import nose

seed(111)
normal_sample = normal(0, 1, 1000000)

def test_autocorr():
    """Test autocorrelation and autocovariance functions"""

    assert_almost_equal(stats.autocorr(normal_sample), 0, 2)

    y = [(normal_sample[i-1] + normal_sample[i])/2 for i in range(1, len(normal_sample))]
    assert_almost_equal(stats.autocorr(y), 0.5, 2)

def test_hpd():
    """Test HPD calculation"""

    interval = stats.hpd(normal_sample)

    assert_array_almost_equal(interval, [-1.96, 1.96], 2)

def test_make_indices():
    """Test make_indices function"""

    ind = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    assert_equal(ind, stats.make_indices((2, 3)))

def test_mc_error():
    """Test batch standard deviation function"""

    x = random(100000)

    assert(stats.mc_error(x) < 0.0025)

def test_quantiles():
    """Test quantiles function"""

    q = stats.quantiles(normal_sample)

    assert_array_almost_equal(sorted(q.values()), [-1.96, -0.67, 0, 0.67, 1.96], 2)


def test_summary_1_value_model():
    mu = -2.1
    tau = 1.3
    with pm.Model() as model:
        x = pm.Normal('x', mu, tau, testval=.1)
        step = pm.Metropolis(model.vars, np.diag([1.]))
        trace = pm.sample(100, step=step)
    stats.summary(trace)


def test_summary_2_value_model():
    mu = -2.1
    tau = 1.3
    with pm.Model() as model:
        x = pm.Normal('x', mu, tau, shape=2, testval=[.1, .1])
        step = pm.Metropolis(model.vars, np.diag([1.]))
        trace = pm.sample(100, step=step)
    stats.summary(trace)


def test_summary_2dim_value_model():
    mu = -2.1
    tau = 1.3
    with pm.Model() as model:
        x = pm.Normal('x', mu, tau, shape=(2, 2),
                   testval=np.tile(.1, (2, 2)))
        step = pm.Metropolis(model.vars, np.diag([1.]))
        trace = pm.sample(100, step=step)

    with warnings.catch_warnings(record=True) as wrn:
        stats.summary(trace)
        assert len(wrn) == 1
        assert str(wrn[0].message) == 'Skipping x (above 1 dimension)'


def test_summary_format_values():
    roundto = 2
    summ = stats._Summary(roundto)
    d = {'nodec': 1, 'onedec': 1.0, 'twodec': 1.00, 'threedec': 1.000}
    summ._format_values(d)
    for val in d.values():
        assert val == '1.00'


def test_stat_summary_format_hpd_values():
    roundto = 2
    summ = stats._StatSummary(roundto, None, 0.05)
    d = {'nodec': 1, 'hpd': [1, 1]}
    summ._format_values(d)
    for key, val in d.items():
        if key == 'hpd':
            assert val == '[1.00, 1.00]'
        else:
            assert val == '1.00'


@nose.tools.raises(IndexError)
def test_calculate_stats_variable_size1_not_adjusted():
    sample = np.arange(10)
    list(stats._calculate_stats(sample, 5, 0.05))


def test_calculate_stats_variable_size1_adjusted():
    sample = np.arange(10)[:, None]
    result_size = len(list(stats._calculate_stats(sample, 5, 0.05)))
    assert result_size == 1

def test_calculate_stats_variable_size2():
    ## 2 traces of 5
    sample = np.arange(10).reshape(5, 2)
    result_size = len(list(stats._calculate_stats(sample, 5, 0.05)))
    assert result_size == 2


@nose.tools.raises(IndexError)
def test_calculate_pquantiles_variable_size1_not_adjusted():
    sample = np.arange(10)
    qlist = (0.25, 25, 50, 75, 0.98)
    list(stats._calculate_posterior_quantiles(sample,
                                              qlist))


def test_calculate_pquantiles_variable_size1_adjusted():
    sample = np.arange(10)[:, None]
    qlist = (0.25, 25, 50, 75, 0.98)
    result_size = len(list(stats._calculate_posterior_quantiles(sample,
                                                                qlist)))
    assert result_size == 1


def test_stats_value_line():
    roundto = 1
    summ = stats._StatSummary(roundto, None, 0.05)
    values = [{'mean': 0, 'sd': 1, 'mce': 2, 'hpd': [4, 4]},
              {'mean': 5, 'sd': 6, 'mce': 7, 'hpd': [8, 8]},]

    expected = ['0.0              1.0              2.0              [4.0, 4.0]',
                '5.0              6.0              7.0              [8.0, 8.0]']
    result = list(summ._create_value_output(values))
    assert result == expected


def test_post_quantile_value_line():
    roundto = 1
    summ = stats._PosteriorQuantileSummary(roundto, 0.05)
    values = [{'lo': 0, 'q25': 1, 'q50': 2, 'q75': 4, 'hi': 5},
              {'lo': 6, 'q25': 7, 'q50': 8, 'q75': 9, 'hi': 10},]

    expected = ['0.0            1.0            2.0            4.0            5.0',
                '6.0            7.0            8.0            9.0            10.0']
    result = list(summ._create_value_output(values))
    assert result == expected


def test_stats_output_lines():
    roundto = 1
    x = np.arange(10).reshape(5, 2)

    summ = stats._StatSummary(roundto, 5, 0.05)

    expected = ['  Mean             SD               MC Error         95% HPD interval',
                '  -------------------------------------------------------------------',
                '  4.0              2.8              1.3              [0.0, 8.0]',
                '  5.0              2.8              1.3              [1.0, 9.0]',]
    result = list(summ._get_lines(x))
    assert result == expected


def test_posterior_quantiles_output_lines():
    roundto = 1
    x = np.arange(10).reshape(5, 2)

    summ = stats._PosteriorQuantileSummary(roundto, 0.05)

    expected = ['  Posterior quantiles:',
                '  2.5            25             50             75             97.5',
                '  |--------------|==============|==============|--------------|',
                '  0.0            2.0            4.0            6.0            8.0',
                '  1.0            3.0            5.0            7.0            9.0']

    result = list(summ._get_lines(x))
    assert result == expected
