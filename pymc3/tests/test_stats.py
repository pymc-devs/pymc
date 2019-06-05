from .models import Model, Normal, Metropolis
import numpy as np
import numpy.testing as npt
import pandas as pd
import pymc3 as pm
from .helpers import SeededTest
from ..tests import backend_fixtures as bf
from ..backends import ndarray
from ..stats import (summary, autocorr, autocov, hpd, mc_error, quantiles,
                     make_indices, bfmi, r2_score)
from ..theanof import floatX_array
import pymc3.stats as pmstats
from numpy.random import random, normal
from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal
from scipy import stats as st
import copy


def test_log_post_trace():
    with pm.Model() as model:
        pm.Normal('y')
        trace = pm.sample(10, tune=10, chains=1)

    logp = pmstats._log_post_trace(trace, model)
    assert logp.shape == (len(trace), 0)

    with pm.Model() as model:
        pm.Normal('a')
        pm.Normal('y', observed=np.zeros((2, 3)))
        trace = pm.sample(10, tune=10, chains=1)

    logp = pmstats._log_post_trace(trace, model)
    assert logp.shape == (len(trace), 6)
    npt.assert_allclose(logp, -0.5 * np.log(2 * np.pi), atol=1e-7)

    with pm.Model() as model:
        pm.Normal('a')
        pm.Normal('y', observed=np.zeros((2, 3)))
        data = pd.DataFrame(np.zeros((3, 4)))
        data.values[1, 1] = np.nan
        pm.Normal('y2', observed=data)
        data = data.copy()
        data.values[:] = np.nan
        pm.Normal('y3', observed=data)
        trace = pm.sample(10, tune=10, chains=1)

    logp = pmstats._log_post_trace(trace, model)
    assert logp.shape == (len(trace), 17)
    npt.assert_allclose(logp, -0.5 * np.log(2 * np.pi), atol=1e-7)


def test_compare():
    np.random.seed(42)
    x_obs = np.random.normal(0, 1, size=100)

    with pm.Model() as model0:
        mu = pm.Normal('mu', 0, 1)
        x = pm.Normal('x', mu=mu, sigma=1, observed=x_obs)
        trace0 = pm.sample(1000)

    with pm.Model() as model1:
        mu = pm.Normal('mu', 0, 1)
        x = pm.Normal('x', mu=mu, sigma=0.8, observed=x_obs)
        trace1 = pm.sample(1000)

    with pm.Model() as model2:
        mu = pm.Normal('mu', 0, 1)
        x = pm.StudentT('x', nu=1, mu=mu, lam=1, observed=x_obs)
        trace2 = pm.sample(1000)

    traces = [trace0, copy.copy(trace0)]
    models = [model0, copy.copy(model0)]

    model_dict = dict(zip(models, traces))

    w_st = pm.compare(model_dict, method='stacking')['weight']
    w_bb_bma = pm.compare(model_dict, method='BB-pseudo-BMA')['weight']
    w_bma = pm.compare(model_dict, method='pseudo-BMA')['weight']

    assert_almost_equal(w_st[0], w_st[1])
    assert_almost_equal(w_bb_bma[0], w_bb_bma[1])
    assert_almost_equal(w_bma[0], w_bma[1])

    assert_almost_equal(np.sum(w_st), 1.)
    assert_almost_equal(np.sum(w_bb_bma), 1.)
    assert_almost_equal(np.sum(w_bma), 1.)

    traces = [trace0, trace1, trace2]
    models = [model0, model1, model2]

    model_dict = dict(zip(models, traces))

    w_st = pm.compare(model_dict, method='stacking')['weight']
    w_bb_bma = pm.compare(model_dict, method='BB-pseudo-BMA')['weight']
    w_bma = pm.compare(model_dict, method='pseudo-BMA')['weight']

    assert(w_st[0] > w_st[1] > w_st[2])
    assert(w_bb_bma[0] > w_bb_bma[1] > w_bb_bma[2])
    assert(w_bma[0] > w_bma[1] > w_bma[2])

    assert_almost_equal(np.sum(w_st), 1.)
    assert_almost_equal(np.sum(w_st), 1.)
    assert_almost_equal(np.sum(w_st), 1.)


class TestStats(SeededTest):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.normal_sample = normal(0, 1, 200000)

    def test_autocorr(self):
        """Test autocorrelation and autocovariance functions"""
        assert_almost_equal(autocorr(self.normal_sample)[1], 0, 2)
        y = [(self.normal_sample[i - 1] + self.normal_sample[i]) /
             2 for i in range(1, len(self.normal_sample))]
        assert_almost_equal(autocorr(np.asarray(y))[1], 0.5, 2)
        lag = 5
        acov_np = np.cov(self.normal_sample[:-lag],
                         self.normal_sample[lag:], bias=1)[0, 1]
        acov_pm = autocov(self.normal_sample)[lag]
        assert_almost_equal(acov_pm, acov_np, 7)

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
            Normal('x', mu, tau, testval=floatX_array(.1))
            step = Metropolis(model.vars, np.diag([1.]), blocked=True)
            trace = pm.sample(100, step=step)
        summary(trace)

    def test_summary_1d_variable_model(self):
        mu = -2.1
        tau = 1.3
        with Model() as model:
            Normal('x', mu, tau, shape=2, testval=floatX_array([.1, .1]))
            step = Metropolis(model.vars, np.diag([1.]), blocked=True)
            trace = pm.sample(100, step=step)
        summary(trace)

    def test_summary_2d_variable_model(self):
        mu = -2.1
        tau = 1.3
        with Model() as model:
            Normal('x', mu, tau, shape=(2, 2),
                   testval=floatX_array(np.tile(.1, (2, 2))))
            step = Metropolis(model.vars, np.diag([1.]), blocked=True)
            trace = pm.sample(100, step=step)
        summary(trace)

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

    def test_bfmi(self):
        trace = {'energy': np.array([1, 2, 3, 4])}

        assert_almost_equal(bfmi(trace), 0.8)

    def test_r2_score(self):
        x = np.linspace(0, 1, 100)
        y = np.random.normal(x, 1)
        res = st.linregress(x, y)
        assert_almost_equal(res.rvalue ** 2,
                            r2_score(y, res.intercept +
                                     res.slope * x).r2_median,
                            2)

class TestDfSummary(bf.ModelBackendSampledTestCase):
    backend = ndarray.NDArray
    name = 'text-db'
    shape = (2, 3)

    def test_column_names(self):
        ds = summary(self.mtrace, batches=3)
        npt.assert_equal(np.array(['mean', 'sd', 'mc_error',
                                   'hpd_2.5', 'hpd_97.5',
                                   'n_eff', 'Rhat']),
                         ds.columns)

    def test_column_names_decimal_hpd(self):
        ds = summary(self.mtrace, batches=3, alpha=0.001)
        npt.assert_equal(np.array(['mean', 'sd', 'mc_error',
                                   'hpd_0.05', 'hpd_99.95',
                                   'n_eff', 'Rhat']),
                         ds.columns)

    def test_column_names_custom_function(self):
        def customf(x):
            return pd.Series(np.mean(x, 0), name='my_mean')

        ds = summary(self.mtrace, batches=3, stat_funcs=[customf])
        npt.assert_equal(np.array(['my_mean']), ds.columns)

    def test_column_names_custom_function_extend(self):
        def customf(x):
            return pd.Series(np.mean(x, 0), name='my_mean')

        ds = summary(self.mtrace, batches=3,
                        stat_funcs=[customf], extend=True)
        npt.assert_equal(np.array(['mean', 'sd', 'mc_error',
                                   'hpd_2.5', 'hpd_97.5', 'my_mean',
                                   'n_eff', 'Rhat']),
                         ds.columns)

    def test_value_alignment(self):
        mtrace = self.mtrace
        ds = summary(mtrace, batches=3)
        for var in mtrace.varnames:
            result = mtrace[var].mean(0)
            for idx, val in np.ndenumerate(result):
                if idx:
                    vidx = var + '__' + '_'.join([str(i) for i in idx])
                else:
                    vidx = var
                npt.assert_equal(val, ds.loc[vidx, 'mean'])

    def test_row_names(self):
        with Model():
            pm.Uniform('x', 0, 1)
            step = Metropolis()
            trace = pm.sample(100, step=step)
        ds = summary(trace, batches=3, include_transformed=True)
        npt.assert_equal(np.array(['x_interval__', 'x']),
                         ds.index)

    def test_value_n_eff_rhat(self):
        mu = -2.1
        tau = 1.3
        with Model():
            Normal('x0', mu, tau, testval=floatX_array(.1)) # 0d
            Normal('x1', mu, tau, shape=2, testval=floatX_array([.1, .1]))# 1d
            Normal('x2', mu, tau, shape=(2, 2),
                   testval=floatX_array(np.tile(.1, (2, 2))))# 2d
            Normal('x3', mu, tau, shape=(2, 2, 3),
                   testval=floatX_array(np.tile(.1, (2, 2, 3))))# 3d
            trace = pm.sample(100, step=pm.Metropolis())
        for varname in trace.varnames:
            # test effective_n value
            n_eff = pm.effective_n(trace, varnames=[varname])[varname]
            n_eff_df = np.asarray(
                    pm.summary(trace, varnames=[varname])['n_eff']
                                 ).reshape(n_eff.shape)
            npt.assert_equal(n_eff, n_eff_df)

            # test Rhat value
            rhat = pm.gelman_rubin(trace, varnames=[varname])[varname]
            rhat_df = np.asarray(
                    pm.summary(trace, varnames=[varname])['Rhat']
                                 ).reshape(rhat.shape)
            npt.assert_equal(rhat, rhat_df)

    def test_psis(self):
        lw = np.random.randn(20000, 10)
        _, ks = pm.stats._psislw(lw, 1.)
        npt.assert_array_less(ks, .5)
