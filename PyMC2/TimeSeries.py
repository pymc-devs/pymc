# encoding: utf-8

# TimeSeries
# Copyright (c) 2006 Chris Fonnesbeck

from numpy import random, squeeze, asarray
from numpy import arange, array, atleast_1d, concatenate, dot, resize
import unittest, pdb

_plotter = None
try:
    from Matplot import PlotFactory
    _plotter = PlotFactory()
except ImportError:
    print 'Matplotlib module not detected ... plotting disabled.'
        
def autocov(series, lag, n_minus_k=False):
    # Sample autocovariance function at specified lag. Use n - k as
    # denominator if n_minus_k flag is true.
    
    n = len(series)
    zbar = series.mean()
    
    return sum([(series[i] - zbar) * (series[i + lag] - zbar) for i in range(n - lag)]) / ((n - lag) * n_minus_k or n)
       
def autocorr(x, lag=1):
    """Sample autocorrelation at specified lag.
    The autocorrelation is the correlation of x_i with x_{i+lag}.
    """
    x = squeeze(asarray(x))
    mu = x.mean()
    v = x.var()
    return ((x[:-lag]-mu)*(x[lag:]-mu)).sum()/v/(len(x) - lag)
    
def correlogram(series, maxlag, name='', plotter=None):
    # Plot correlogram up to specified maximum lag
    
    plotter = plotter or _plotter
    
    plotter.bar_series_plot({name + ' correlogram': [autocorr(series, k) for k in range(maxlag + 1)]}, ylab='Autocorrelation')
    
def partial_autocorr(series, lag):
    # Partial autocorrelation function, using Durbin (1960)
    # recursive algorithm
    
    # Initialize matrices of phi and rho
    phi = resize(0.0, (lag, lag))
    rho = resize(0.0, lag)
    
    # \phi_{1,1} = \rho_1
    phi[0, 0] = rho[0] = autocorr(series, 1)
    
    for k in range(1, lag):
        
        # Calculate autocorrelation for current lag
        rho[k] = autocorr(series, k + 1)
        
        for j in range(k - 1):
            
            # \phi_{k+1,j} = \phi_{k,j} - \phi_{k+1,k+1} * \phi_{k,k+1-j}
            phi[k - 1, j] = phi[k - 2, j] - phi[k - 1, k - 1] * phi[k - 2, k - 2 - j]
        
        # Numerator: \rho_{k+1} - \sum_{j=1}^k \phi_{k,j}\rho_j
        phi[k, k] = rho[k] - sum([phi[k - 1, j] * rho[k - 1 - j] for j in range(k)])
        
        # Denominator: 1 - \sum_{j=1}^k \phi_{k,j}\rho_j
        phi[k, k] /= 1 - sum([phi[k - 1, j] * rho[j] for j in range(k)])
    
    # Return partial autocorrelation value
    return phi[lag - 1, lag - 1]
    
"""
def ar_process(length, stochs=[1.], mu=0., dist='normal', scale=1):
    # Generate AR(p) process of given length, where p=len(stochs).
    
    # Initialize series with mean value
    series = resize(float(mu), length)
    
    # Enforce array type for stochs
    stochs = atleast_1d(stochs)
    
    # Degree of process
    p = len(stochs)
    
    # Specify error distribution
    if dist is 'normal':
        a = random.normal(0, scale, length)
    elif dist is 'cauchy':
        a = random.standard_cauchy(length) * scale
    elif dist is 't':
        a = random.standard_t(scale, length)
    else:
        print 'Invalid error disitrbution'
        return
    
    # Generate autoregressive series
    for t in range(1, length):
        series[t] = dot(stochs[max(p-t, 0):], series[t - min(t, p):t] - mu) + a[t] + mu
        
    return series
    
def ma_process(length, stochs=[1.], mu=0., dist='normal', scale=1):
    # Generate MA(q) process of given length, where q=len(stochs).
    
    # Enforce array type for stochs
    stochs = concatenate(([1], -1 * atleast_1d(stochs))).tolist()
    # Reverse order of stochs for calculations below
    stochs.reverse()
    
    # Degree of process
    q = len(stochs) - 1
    
    # Specify error distribution
    if dist is 'normal':
        a = random.normal(0, scale, length)
    elif dist is 'cauchy':
        a = random.standard_cauchy(length) * scale
    elif dist is 't':
        a = random.standard_t(scale, length)
    else:
        print 'Invalid error disitrbution'
        return
    
    # Generate moving average series
    series = array([mu + dot(stochs[max(q - t + 1, 0):], a[t - min(t, q + 1):t]) for t in range(1, length)])
    
    return series
"""
    
def arma_process(length, ar_stochs=[1.], ma_stochs=[1.], mu=0., dist='normal', scale=1):
    """ Generate ARMA(p,q) process of given length, where p=len(ar_stochs) and q=len(ma_stochs)."""
    
    # Initialize series with mean value
    series = resize(float(mu), length)
    
    # Enforce array type for stochs
    ar_stochs = atleast_1d(ar_stochs)
    ma_stochs = concatenate(([1], -1 * atleast_1d(ma_stochs))).tolist()
    # Reverse order of stochs for calculations below
    ma_stochs.reverse()
    
    # Degree of process
    p, q = len(ar_stochs), len(ma_stochs) - 1
    
    # Specify error distribution
    if dist is 'normal':
        a = random.normal(0, scale, length)
    elif dist is 'cauchy':
        a = random.standard_cauchy(length) * scale
    elif dist is 't':
        a = random.standard_t(scale, length)
    else:
        print 'Invalid error disitrbution'
        return
    
    # Generate autoregressive series
    for t in range(1, length):
        
        # Autoregressive piece
        series[t] += dot(ar_stochs[max(p-t, 0):], series[t - min(t, p):t] - mu)
        
        # Moving average piece
        series[t] += dot(ma_stochs[max(q - t + 1, 0):], a[t - min(t, q + 1):t])
        
    return series
    
def ar_process(length, stochs=[1.], mu=0., dist='normal', scale=1):
    """Generate AR(p) process of given length, where p=len(stochs)."""
    
    return arma_process(length, ar_stochs=stochs, ma_stochs=[], mu=mu, dist=dist, scale=scale)
    
def ma_process(length, stochs=[1.], mu=0., dist='normal', scale=1):
    """Generate MA(q) process of given length, where q=len(stochs)."""

    return arma_process(length, ar_stochs=[], ma_stochs=stochs, mu=mu, dist=dist, scale=scale)


class TimeSeriesTests(unittest.TestCase):
    
    def setUp(self):
        
        # Sample iid normal time series
        self.ts = array(random.normal(size=20))
        
    def testAutocovariance(self):
        # Autocovariance tests
        
        n = len(self.ts)

        # Confirm that covariance at lag 0 equals variance
        self.assertAlmostEqual(autocov(self.ts, 0), self.ts.var(), 10, "Covariance at lag 0 not equal to variance")
        
        self.failIf(sum([self.ts.var() < autocov(self.ts, k) for k in range(1, n)]), "All covariances not less than or equal to variance")
        
    def testARIMA(self):
        # Test ARIMA estimation
        
        pass
        

if __name__ == '__main__':
    unittest.main()
