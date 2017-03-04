from .models import Model, Normal, Metropolis
import numpy as np
import numpy.testing as npt
import pandas as pd
import pymc3 as pm
from .helpers import SeededTest
from ..tests import backend_fixtures as bf
from ..backends import ndarray
from ..stats import df_summary, autocorr, hpd, mc_error, quantiles, make_indices
from numpy.random import random, normal
from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal
from scipy import stats as st


class TestStats(SeededTest):
    @classmethod
    def setUpClass(cls):
        super(TestStats, cls).setUpClass()
        cls.normal_sample = normal(0, 1, 200000)

    def test_autocorr(self):
        """Test autocorrelation and autocovariance functions"""
        assert_almost_equal(autocorr(self.normal_sample), 0, 2)
        y = [(self.normal_sample[i - 1] + self.normal_sample[i]) /
             2 for i in range(1, len(self.normal_sample))]
        assert_almost_equal(autocorr(y), 0.5, 2)

    def test_dic(self):
        """Test deviance information criterion calculation"""
        x_obs = np.arange(6)

        with pm.Model():
            p = pm.Beta('p', 1., 1., transform=None)
            pm.Binomial('x', 5, p, observed=x_obs)

            step = pm.Metropolis()
            trace = pm.sample(100, step)
            calculated = pm.dic(trace)

        mean_deviance = -2 * st.binom.logpmf(
            np.repeat(np.atleast_2d(x_obs), 100, axis=0),
            5,
            np.repeat(np.atleast_2d(trace['p']), 6, axis=0).T).sum(axis=1).mean()
        deviance_at_mean = -2 * st.binom.logpmf(x_obs, 5, trace['p'].mean()).sum()
        actual = 2 * mean_deviance - deviance_at_mean

        assert_almost_equal(calculated, actual, decimal=2)

    def test_bpic(self):
        """Test Bayesian predictive information criterion"""
        x_obs = np.arange(6)

        with pm.Model():
            p = pm.Beta('p', 1., 1., transform=None)
            pm.Binomial('x', 5, p, observed=x_obs)

            step = pm.Metropolis()
            trace = pm.sample(100, step)
            calculated = pm.bpic(trace)

        mean_deviance = -2 * st.binom.logpmf(
            np.repeat(np.atleast_2d(x_obs), 100, axis=0),
            5,
            np.repeat(np.atleast_2d(trace['p']), 6, axis=0).T).sum(axis=1).mean()
        deviance_at_mean = -2 * st.binom.logpmf(x_obs, 5, trace['p'].mean()).sum()
        actual = 3 * mean_deviance - 2 * deviance_at_mean

        assert_almost_equal(calculated, actual, decimal=2)

    def test_waic(self):
        """Test widely available information criterion calculation"""
        x_obs = np.arange(6)

        with pm.Model():
            p = pm.Beta('p', 1., 1., transform=None)
            pm.Binomial('x', 5, p, observed=x_obs)

            step = pm.Metropolis()
            trace = pm.sample(100, step)
            calculated_waic = pm.waic(trace)

        log_py = st.binom.logpmf(np.atleast_2d(x_obs).T, 5, trace['p']).T

        lppd_i = np.log(np.mean(np.exp(log_py), axis=0))
        vars_lpd = np.var(log_py, axis=0)
        waic_i = - 2 * (lppd_i - vars_lpd)

        actual_waic_se = np.sqrt(len(waic_i) * np.var(waic_i))
        actual_waic = np.sum(waic_i)

        assert_almost_equal(calculated_waic.WAIC, actual_waic, decimal=2)
        assert_almost_equal(calculated_waic.WAIC_se, actual_waic_se, decimal=2)

    def test_hpd(self):
        """Test HPD calculation"""
        interval = hpd(self.normal_sample)
        assert_array_almost_equal(interval, [-1.96, 1.96], 2)

    def test_make_indices(self):
        """Test make_indices function"""
        ind = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        assert_equal(ind, make_indices((2, 3)))

    def test_mc_error(self):
        """Test batch standard deviation function"""
        assert(mc_error(random(100000) < 0.0025))

    def test_quantiles(self):
        """Test quantiles function"""
        q = quantiles(self.normal_sample)
        assert_array_almost_equal(sorted(q.values()), [-1.96, -0.67, 0, 0.67, 1.96], 2)

    # For all the summary tests, the number of dimensions refer to the
    # original variable dimensions, not the MCMC trace dimensions.
    def test_summary_0d_variable_model(self):
        mu = -2.1
        tau = 1.3
        with Model() as model:
            Normal('x', mu, tau, testval=.1)
            step = Metropolis(model.vars, np.diag([1.]), blocked=True)
            trace = pm.sample(100, step=step)
        pm.summary(trace)

    def test_summary_1d_variable_model(self):
        mu = -2.1
        tau = 1.3
        with Model() as model:
            Normal('x', mu, tau, shape=2, testval=[.1, .1])
            step = Metropolis(model.vars, np.diag([1.]), blocked=True)
            trace = pm.sample(100, step=step)
        pm.summary(trace)

    def test_summary_2d_variable_model(self):
        mu = -2.1
        tau = 1.3
        with Model() as model:
            Normal('x', mu, tau, shape=(2, 2),
                   testval=np.tile(.1, (2, 2)))
            step = Metropolis(model.vars, np.diag([1.]), blocked=True)
            trace = pm.sample(100, step=step)
        pm.summary(trace)

    def test_summary_format_values(self):
        roundto = 2
        summ = pm.stats._Summary(roundto)
        d = {'nodec': 1, 'onedec': 1.0, 'twodec': 1.00, 'threedec': 1.000}
        summ._format_values(d)
        for val in d.values():
            assert val == '1.00'

    def test_stat_summary_format_hpd_values(self):
        roundto = 2
        summ = pm.stats._StatSummary(roundto, None, 0.05)
        d = {'nodec': 1, 'hpd': [1, 1]}
        summ._format_values(d)
        for key, val in d.items():
            if key == 'hpd':
                assert val == '[1.00, 1.00]'
            else:
                assert val == '1.00'

    def test_calculate_stats_0d_variable(self):
        sample = np.arange(10)
        result = list(pm.stats._calculate_stats(sample, 5, 0.05))
        assert result[0] == ()
        assert len(result) == 2

    def test_calculate_stats_variable_1d_variable(self):
        sample = np.arange(10).reshape(5, 2)
        result = list(pm.stats._calculate_stats(sample, 5, 0.05))
        assert result[0] == ()
        assert len(result) == 3

    def test_calculate_pquantiles_0d_variable(self):
        sample = np.arange(10)[:, None]
        qlist = (0.25, 25, 50, 75, 0.98)
        result = list(pm.stats._calculate_posterior_quantiles(sample, qlist))
        assert result[0] == ()
        assert len(result) == 2

    def test_stats_value_line(self):
        roundto = 1
        summ = pm.stats._StatSummary(roundto, None, 0.05)
        values = [{'mean': 0, 'sd': 1, 'mce': 2, 'hpd': [4, 4]},
                  {'mean': 5, 'sd': 6, 'mce': 7, 'hpd': [8, 8]}, ]

        expected = ['0.0              1.0              2.0              [4.0, 4.0]',
                    '5.0              6.0              7.0              [8.0, 8.0]']
        result = list(summ._create_value_output(values))
        assert result == expected

    def test_post_quantile_value_line(self):
        roundto = 1
        summ = pm.stats._PosteriorQuantileSummary(roundto, 0.05)
        values = [{'lo': 0, 'q25': 1, 'q50': 2, 'q75': 4, 'hi': 5},
                  {'lo': 6, 'q25': 7, 'q50': 8, 'q75': 9, 'hi': 10}, ]

        expected = ['0.0            1.0            2.0            4.0            5.0',
                    '6.0            7.0            8.0            9.0            10.0']
        result = list(summ._create_value_output(values))
        assert result == expected

    def test_stats_output_lines_0d_variable(self):
        roundto = 1
        x = np.arange(5)
        summ = pm.stats._StatSummary(roundto, 5, 0.05)
        expected = ['  Mean             SD               MC Error         95% HPD interval',
                    '  -------------------------------------------------------------------',
                    '  ',
                    '  2.0              1.4              0.6              [0.0, 4.0]', ]

        result = list(summ._get_lines(x))
        assert result == expected

    def test_stats_output_lines_1d_variable(self):
        roundto = 1
        x = np.arange(10).reshape(5, 2)
        summ = pm.stats._StatSummary(roundto, 5, 0.05)
        expected = ['  Mean             SD               MC Error         95% HPD interval',
                    '  -------------------------------------------------------------------',
                    '  ',
                    '  4.0              2.8              1.3              [0.0, 8.0]',
                    '  5.0              2.8              1.3              [1.0, 9.0]', ]
        result = list(summ._get_lines(x))
        assert result == expected

    def test_stats_output_lines_2d_variable(self):
        roundto = 1
        x = np.arange(20).reshape(5, 2, 2)
        summ = pm.stats._StatSummary(roundto, 5, 0.05)
        expected = ['  Mean             SD               MC Error         95% HPD interval',
                    '  -------------------------------------------------------------------',
                    '  ..............................[0, :]...............................',
                    '  8.0              5.7              2.5              [0.0, 16.0]',
                    '  9.0              5.7              2.5              [1.0, 17.0]',
                    '  ..............................[1, :]...............................',
                    '  10.0             5.7              2.5              [2.0, 18.0]',
                    '  11.0             5.7              2.5              [3.0, 19.0]', ]
        result = list(summ._get_lines(x))
        assert result == expected

    def test_stats_output_HPD_interval_format(self):
        roundto = 1
        x = np.arange(5)
        summ = pm.stats._StatSummary(roundto, 5, 0.05)
        expected = '  Mean             SD               MC Error         95% HPD interval'
        result = list(summ._get_lines(x))
        assert result[0] == expected

        summ = pm.stats._StatSummary(roundto, 5, 0.001)
        expected = '  Mean             SD               MC Error         99.9% HPD interval'
        result = list(summ._get_lines(x))
        assert result[0] == expected

    def test_posterior_quantiles_output_lines_0d_variable(self):
        roundto = 1
        x = np.arange(5)
        summ = pm.stats._PosteriorQuantileSummary(roundto, 0.05)
        expected = ['  Posterior quantiles:',
                    '  2.5            25             50             75             97.5',
                    '  |--------------|==============|==============|--------------|',
                    '  ',
                    '  0.0            1.0            2.0            3.0            4.0', ]

        result = list(summ._get_lines(x))
        assert result == expected

    def test_posterior_quantiles_output_lines_1d_variable(self):
        roundto = 1
        x = np.arange(10).reshape(5, 2)
        summ = pm.stats._PosteriorQuantileSummary(roundto, 0.05)
        expected = ['  Posterior quantiles:',
                    '  2.5            25             50             75             97.5',
                    '  |--------------|==============|==============|--------------|',
                    '  ',
                    '  0.0            2.0            4.0            6.0            8.0',
                    '  1.0            3.0            5.0            7.0            9.0']

        result = list(summ._get_lines(x))
        assert result == expected

    def test_posterior_quantiles_output_lines_2d_variable(self):
        roundto = 1
        x = np.arange(20).reshape(5, 2, 2)
        summ = pm.stats._PosteriorQuantileSummary(roundto, 0.05)
        expected = ['  Posterior quantiles:',
                    '  2.5            25             50             75             97.5',
                    '  |--------------|==============|==============|--------------|',
                    '  .............................[0, :].............................',
                    '  0.0            4.0            8.0            12.0           16.0',
                    '  1.0            5.0            9.0            13.0           17.0',
                    '  .............................[1, :].............................',
                    '  2.0            6.0            10.0           14.0           18.0',
                    '  3.0            7.0            11.0           15.0           19.0', ]

        result = list(summ._get_lines(x))
        assert result == expected

    def test_groupby_leading_idxs_0d_variable(self):
        result = {k: list(v) for k, v in pm.stats._groupby_leading_idxs(())}
        assert list(result.keys()) == [()]
        assert result[()] == [()]

    def test_groupby_leading_idxs_1d_variable(self):
        result = {k: list(v) for k, v in pm.stats._groupby_leading_idxs((2,))}
        assert list(result.keys()) == [()]
        assert result[()] == [(0,), (1,)]

    def test_groupby_leading_idxs_2d_variable(self):
        result = {k: list(v) for k, v in pm.stats._groupby_leading_idxs((2, 3))}
        expected_keys = [(0,), (1,)]
        keys = list(result.keys())
        assert len(keys) == len(expected_keys)
        for key in keys:
            assert result[key] == [key + (0,), key + (1,), key + (2,)]

    def test_groupby_leading_idxs_3d_variable(self):
        result = {k: list(v) for k, v in pm.stats._groupby_leading_idxs((2, 3, 2))}

        expected_keys = [(0, 0), (0, 1), (0, 2),
                         (1, 0), (1, 1), (1, 2)]
        keys = list(result.keys())
        assert len(keys) == len(expected_keys)
        for key in keys:
            assert result[key] == [key + (0,), key + (1,)]


class TestDfSummary(bf.ModelBackendSampledTestCase):
    backend = ndarray.NDArray
    name = 'text-db'
    shape = (2, 3)

    def test_column_names(self):
        ds = df_summary(self.mtrace, batches=3)
        npt.assert_equal(np.array(['mean', 'sd', 'mc_error',
                                   'hpd_2.5', 'hpd_97.5']),
                         ds.columns)

    def test_column_names_decimal_hpd(self):
        ds = df_summary(self.mtrace, batches=3, alpha=0.001)
        npt.assert_equal(np.array(['mean', 'sd', 'mc_error',
                                   'hpd_0.05', 'hpd_99.95']),
                         ds.columns)

    def test_column_names_custom_function(self):
        def customf(x):
            return pd.Series(np.mean(x, 0), name='my_mean')

        ds = df_summary(self.mtrace, batches=3, stat_funcs=[customf])
        npt.assert_equal(np.array(['my_mean']), ds.columns)

    def test_column_names_custom_function_extend(self):
        def customf(x):
            return pd.Series(np.mean(x, 0), name='my_mean')

        ds = df_summary(self.mtrace, batches=3,
                        stat_funcs=[customf], extend=True)
        npt.assert_equal(np.array(['mean', 'sd', 'mc_error',
                                   'hpd_2.5', 'hpd_97.5', 'my_mean']),
                         ds.columns)

    def test_value_alignment(self):
        mtrace = self.mtrace
        ds = df_summary(mtrace, batches=3)
        for var in mtrace.varnames:
            result = mtrace[var].mean(0)
            for idx, val in np.ndenumerate(result):
                if idx:
                    vidx = var + '__' + '_'.join([str(i) for i in idx])
                else:
                    vidx = var
                npt.assert_equal(val, ds.loc[vidx, 'mean'])

    def test_row_names(self):
        with Model() as model:
            pm.Uniform('x', 0, 1)
            step = Metropolis()
            trace = pm.sample(100, step=step)
        ds = df_summary(trace, batches=3, include_transformed=True)
        npt.assert_equal(np.array(['x_interval_', 'x']),
                         ds.index)
