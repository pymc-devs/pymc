# encoding: utf-8

# TimeSeries
# Copyright (c) 2006 Chris Fonnesbeck

from numpy import random
from numpy import arange, array, asarray, atleast_1d, concatenate, dot, resize, squeeze
import unittest, pdb

_plotter = None
try:
    from Matplot import PlotFactory
    _plotter = PlotFactory()
except ImportError:
    print 'Matplotlib module not detected ... plotting disabled.'
        
def autocov(x, lag, n_minus_k=False):
    # Sample autocovariance function at specified lag. Use n - k as
    # denominator if n_minus_k flag is true.
    
    x = squeeze(asarray(x))
    mu = x.mean()
    
    if not lag:
        return x.var()
    
    return ((x[:-lag] - mu) * (x[lag:] - mu)).sum() / (n_minus_k * (len(x) - lag) or len(x))
    
def autocorr(x, lag, n_minus_k=False):
    # Sample autocorrelation at specified lag. Use n - k as
    # denominator if n_minus_k flag is true.
    # The autocorrelation is the correlation of x_i with x_{i+lag}.
    
    if not lag:
        return 1
    
    x = squeeze(asarray(x))
    mu = x.mean()
    v = x.var()
    
    return ((x[:-lag] - mu) * (x[lag:] - mu)).sum() / v / (n_minus_k * (len(x) - lag) or len(x))
    
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
def ar_process(length, params=[1.], mu=0., dist='normal', scale=1):
    # Generate AR(p) process of given length, where p=len(params).
    
    # Initialize series with mean value
    series = resize(float(mu), length)
    
    # Enforce array type for parameters
    params = atleast_1d(params)
    
    # Degree of process
    p = len(params)
    
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
        series[t] = dot(params[max(p-t, 0):], series[t - min(t, p):t] - mu) + a[t] + mu
        
    return series
    
def ma_process(length, params=[1.], mu=0., dist='normal', scale=1):
    # Generate MA(q) process of given length, where q=len(params).
    
    # Enforce array type for parameters
    params = concatenate(([1], -1 * atleast_1d(params))).tolist()
    # Reverse order of parameters for calculations below
    params.reverse()
    
    # Degree of process
    q = len(params) - 1
    
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
    series = array([mu + dot(params[max(q - t + 1, 0):], a[t - min(t, q + 1):t]) for t in range(1, length)])
    
    return series
"""
    
def arma_process(length, ar_params=[1.], ma_params=[1.], mu=0., dist='normal', scale=1):
    """ Generate ARMA(p,q) process of given length, where p=len(ar_params) and q=len(ma_params)."""
    
    # Initialize series with mean value
    series = resize(float(mu), length)
    
    # Enforce array type for parameters
    ar_params = atleast_1d(ar_params)
    ma_params = concatenate(([1], -1 * atleast_1d(ma_params))).tolist()
    # Reverse order of parameters for calculations below
    ma_params.reverse()
    
    # Degree of process
    p, q = len(ar_params), len(ma_params) - 1
    
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
        series[t] += dot(ar_params[max(p-t, 0):], series[t - min(t, p):t] - mu)
        
        # Moving average piece
        series[t] += dot(ma_params[max(q - t + 1, 0):], a[t - min(t, q + 1):t])
        
    return series
    
def ar_process(length, params=[1.], mu=0., dist='normal', scale=1):
    """Generate AR(p) process of given length, where p=len(params)."""
    
    return arma_process(length, ar_params=params, ma_params=[], mu=mu, dist=dist, scale=scale)
    
def ma_process(length, params=[1.], mu=0., dist='normal', scale=1):
    """Generate MA(q) process of given length, where q=len(params)."""

    return arma_process(length, ar_params=[], ma_params=params, mu=mu, dist=dist, scale=scale)


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