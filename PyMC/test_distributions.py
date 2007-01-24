"""Test the distributions from flib and their random generators to check
they are consistent with one another.
For each distribution:
1. Select parameters (params = {'alpha':.., 'beta'=...}).
2. Generate N random samples.
3. Compute likelihood for a vector of x values.
4. Compute histogram of samples.
5. Compare histogram with likelihood.
6. Assert if they fit or not.

David- constrain() in flib is giving me bus errors, try for instance
uncommenting the constrain lines in poisson().
-A
Well, you're not calling it correctly, and g77 doesn't understand INFINITY.
At first, I wanted to constrain everything from fortran, but g77 doesn't
really handle exceptions. I finally wrapped the fortran constrain and called
it from python. We'll need a brainstorm on this. One idea is to use f2py for this.
There is some checking that can be done using the check argument. Maybe we can
insert C code to do a loop on each argument and check the constraints.
With pure fortran, the only way I see out of it is to return a very large
negative number to approximate -inf. For the time being, let's leave the
constraining to python.
-D
"""

# TODO: Improve the assertion of consistency.
#       Maybe compare the relative error (hist-like)/like. Doesn't work so well.
#       Tried (hist-like)/sqrt(like), seems to work better. 


from decorators import *
import unittest
import flib
from numpy.testing import *
import numpy as np
from numpy import exp
import utils
PLOT=True
try:
    from scipy import integrate
    SP = True
except:
    print 'You really should install SciPy. Some of the tests might not pass.'
    SP= False
try:
    import pylab as P
except:
    print 'Plotting disabled'
    PLOT=False
SP = False
def consistency(random, like, params, nbins=10, nrandom=1000, nintegration=15,\
    range=None, plot=None):
    """Check the random generator is consistent with the likelihood.

    :Parameters:
      - `random`: function: Random generator.
      - `like`: function: Log probability.
      - `params`: dict:  Parameters for the distribution.
      - `nbins`: int: Number of bins in histogram.
      - `nrandom`: int: Number of random samples.
      - `nintegration`: int: Number of divisions in each bin.
      - `range`: (float,float): Range of histogram.

    :Return (hist, like):
      - `hist`: Histogram of random samples from random(**params).
      - `like`: integrated likelihood over histogram bins.
    """
    # Samples values and compute histogram.
    samples = []
    for i in np.arange(nrandom):
        samples.append(random(**params))
    samples = np.array(samples)
    
    hist, output = utils.histogram(samples, range=range, bins=nbins, normed=True)

    # Compute likelihood along x axis.
    if range is None:
        range = samples.min(), samples.max()
    X = np.linspace(range[0], range[1], nbins*nintegration)
    l = []
    for x in X:
        l.append(like(x, **params))
    L = np.exp(np.array(l))
    
    figuredata = {'samples':samples, 'bins':output['edges'][:-1], \
        'like':L.copy(), 'x':X}
    
    L = L.reshape(nbins, nintegration)
    like = L.mean(1)
    return hist, like, figuredata

def mv_consistency(random, like, params, nbins=10, nrandom=1000, nintegration=15,\
    range=None, plot=None):
    samples = random(n=nrandom, **params)
    hist, edges = np.histogramdd(samples.T, nbins, range, True)
    z = []
    for s in samples:
        z.append(like(s, **params))
    P = np.exp(np.array(z))

def compare_hist(samples, bins, like, x, figname):
    """Plot and save a figure comparing the histogram with the 
    probability.
    
    :Parameters:
      - `samples`: random variables.
      - `like`: probability values.
      - `bins`: histogram bins. 
      - `x`: values at which like is computed. 
    """
    ax = P.subplot(111)
    ax.hist(samples, bins=bins,normed=True, alpha=.5)
    ax.plot(x, like)
    P.savefig('figs/' + figname)
    P.close() 
    
def normalization(like, params, domain, N=100):
    """Integrate the distribution over domain.
    
    :Parameters:
      - `like`: log probability density.
      - `params`: {}: distribution parameters.
      - `domain`: domain of integration. 
      - `N`:  Number of samples for trapezoidal integration.
      
    Note:
      The integration is performed using scipy.integrate.quadg if available.
      Otherwise, a trapezoidal integration is done. 
    """
    f = lambda x: exp(like(x, **params))
    if SP:
        out = integrate.quad(f, domain[0], domain[1])
        return out[0]
    else:
        y = []
        x = np.linspace(domain[0], domain[1], N)
        for i in x:
            y.append(f(i))
        return np.trapz(y,x)
            
    
class test_bernoulli(NumpyTestCase):
    def check_consistency(self):
        N = 5000
        params = {'p':.6}
        samples = []
        for i in range(N):
            samples.append(rbernoulli(**params))
        H = np.bincount(samples)*1./N
        l0 = exp(flib.bernoulli(0, **params))
        l1 = exp(flib.bernoulli(1, **params))
        assert_array_almost_equal(H, [l0,l1], 2)
        # Check normalization
        assert_almost_equal(l0+l1, 1, 4)
        
    def test_calling(self):
        a = flib.bernoulli([0,1,1,0], .4)
        b = flib.bernoulli([0,1,1,0], [.4, .4, .4, .4])
        assert_array_equal(b,a)

class test_beta(NumpyTestCase):
    def check_consistency(self):
        params ={'alpha':3, 'beta':5}
        hist, like, figdata = consistency(rbeta, flib.beta_like, params, nrandom=5000, range=[0,1])
        assert_array_almost_equal((hist-like)/sqrt(like),0,1)
        if PLOT:
            compare_hist(figname='beta', **figdata)

    def test_calling(self):
        a = flib.beta_like([.3,.4,.5], 2,3)
        b = flib.beta_like([.3,.4,.5], [2,2,2],[3,3,3])
        assert_array_equal(a,b)

    def check_normalization(self):
        params ={'alpha':3, 'beta':5}
        integral = normalization(flib.beta_like, params, [0,1], 200)
        assert_almost_equal(integral, 1, 3)
        
class test_binomial(NumpyTestCase):
    def check_consistency(self):
        params={'n':7, 'p':.7}
        N = 2000
        samples =[]
        for i in range(N):
            samples.append(rbinomial(**params))
        H = np.bincount(samples)*1./N
        like = []
        for i in range(params['n']+1):
            like.append(exp(flib.binomial(i, **params)))
        like = np.array(like)
        assert_array_almost_equal(H, like,1)
        # Check_normalization
        assert_almost_equal(like.sum(), 1, 4)

    def test_calling(self):
        a = flib.binomial([3,4,5], 7, .7)
        b = flib.binomial([3,4,5], [7,7,7], .7)
        c = flib.binomial([3,4,5], [7,7,7], [.7,.7,.7])
        assert_equal(a,b)
        assert_equal(a,c)

            
class test_cauchy(NumpyTestCase):
    def check_consistency(self):
        params={'alpha':0, 'beta':.5}
        hist, like, figdata = consistency(rcauchy, flib.cauchy, params, \
        nrandom=5000, range=[-10,10], nbins=21)
        if PLOT:
            compare_hist(figname='cauchy', **figdata)
        assert_array_almost_equal(hist, like,1)
        
            
    def test_calling(self):
        a = flib.cauchy([3,4], 2, 6)
        b = flib.cauchy([3,4], [2,2], [6,6])
        assert_equal(a,b)

    def check_normalization(self):
        params={'alpha':0, 'beta':.5}
        integral = normalization(flib.cauchy, params, [-100,100], 600)
        assert_almost_equal(integral, 1, 2)

class test_chi2(NumpyTestCase):
    """Based on flib.gamma, so no need to make the calling check and 
    normalization check."""
    def check_consistency(self):
        params = {'df':2}
        hist, like, figdata = consistency(rchi2, chi2_like, params, range=[0,15])
        if PLOT:
            compare_hist(figname='chi2', **figdata)
        assert_array_almost_equal(hist, like, 1)
        
class test_dirichlet(NumpyTestCase):
    """Multivariate Dirichlet distribution"""
    def check_random(self):
        theta = np.array([2.,3.])
        r = rdirichlet(theta, n=2000)
        s = theta.sum()
        m = r.mean(0)
        cov_ex = np.cov(r.T)
        
        # Theoretical mean
        M = theta/s
        # Theoretical covariance
        cov_th = -np.outer(theta, theta)/s**2/(s+1.)
        
        assert_array_almost_equal(m,M, 2)
        assert_array_almost_equal(cov_ex, cov_th,1)
        
    def check_like(self):
        theta = np.array([2.,3.])
        x = [4.,2]
        l = flib.dirichlet(x, theta)
        f = utils.dirichlet(x, theta)
        assert_almost_equal(l, sum(np.log(f)), 5)

class test_exponential(NumpyTestCase):
    def check_consistency(self):
        params={'beta':4}
        hist, like, figdata = consistency(rexponential, exponential_like, params,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='exponential', **figdata)
        assert_array_almost_equal(hist, like,1)
        
       
class test_gamma(NumpyTestCase):
    def check_consistency(self):
        params={'alpha':3, 'beta':2}
        hist, like, figdata = consistency(rgamma, flib.gamma, params,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='gamma', **figdata)
        assert_array_almost_equal(hist, like,1)

    def test_calling(self):
        a = flib.gamma([4,5],3,2)
        b = flib.gamma([4,5], [3,3],[2,2])
        assert_equal(a,b)
        
    def check_normalization(self):
        params={'alpha':3, 'beta':2}
        integral = normalization(flib.gamma, params, [.01,20], 200)
        assert_almost_equal(integral, 1, 2)
        
class test_geometric(NumpyTestCase):
    pass

        
class test_poisson(NumpyTestCase):
    def check_consistency(self):
        params = {'mu':2.}
        hist,like, figdata = consistency(rpoisson, flib.poisson, params, nrandom=5000, range=[0,10])
        if PLOT:
            compare_hist(figname='poisson', **figdata)
        assert_array_almost_equal(hist, like,1)
    
    def normalization(self):
        params = {'mu':2.}
        integral = normalization(flib.poisson, params, [0.1, 20], 200)
        assert_almost_equal(integral, 1, 2)
        
    def test_calling(self):
        a = flib.poisson([1,2,3], 2)
        b = flib.poisson([1,2,3], [2,2,2])
        assert_equal(a,b)
        
class test_weibull(NumpyTestCase):
    def check_consistency(self):
        params = {'alpha': 2., 'beta': 3.}
        hist,like, figdata = consistency(rweibull, flib.weibull, params, nrandom=5000, range=[0,10])
        if PLOT:
            compare_hist(figname='weibull', **figdata)
        assert_array_almost_equal(hist, like,1)

    def check_norm(self):
        params = {'alpha': 2., 'beta': 3.}
        integral = normalization(flib.weibull, params, [0,10], N=200)
        assert_almost_equal(integral, 1, 2)
        
    def test_calling(self):
        a = flib.weibull([1,2], 2,3)
        b = flib.weibull([1,2], [2,2], [3,3])
        assert_equal(a,b)




"""
Hyperg is parametrized differently in flib than is hypergeometric in numpy.random.
numpy.random: good, bad, sample
flib: red (=bad), d (=sample), total (=good+bad)

What to do about this?
Either change the argument names in flib or in the distributions wrapper.
-D

class test_hyperg(NumpyTestCase):
    def __init__(self):
        from np.random import poisson
        params = {'ngood':2, 'nbad':5, 'nsample':3}
        hist,like, figdata = consistency(poisson, flib.poisson, params, nrandom=5000, range=[0,10])
"""



if __name__ == '__main__':
    NumpyTest().run()
