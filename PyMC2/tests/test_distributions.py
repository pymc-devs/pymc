"""Test the distributions from flib and their random generators to check
they are consistent with one another.
For each distribution:
1. Select parameters (params = {'alpha':.., 'beta'=...}).
2. Generate N random samples.
3. Compute likelihood for a vector of x values.
4. Compute histogram of samples.
5. Compare histogram with likelihood.
6. Assert if they fit or not.

"""

# TODO: Improve the assertion of consistency.
#       Maybe compare the relative error (hist-like)/like. Doesn't work so well.
#       Tried (hist-like)/sqrt(like), seems to work better.


#from decorators import *
from PyMC2.distributions import *
import unittest
from numpy.testing import *
from PyMC2 import flib, utils
import numpy as np
from numpy import exp, array, cov, prod
import os
PLOT=True
if PLOT is True:
    try:
        os.mkdir('figs')
    except:
        pass
try:
    from scipy import integrate
    from scipy.stats import genextreme
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
def consistency(randomf, likef, params, nbins=10, nrandom=1000, nintegration=15,\
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
    samples = randomf(size=nrandom, **params)

    hist, output = utils.histogram(samples, range=range, bins=nbins, normed=True)

    # Compute likelihood along x axis.
    if range is None:
        range = samples.min(), samples.max()
    x = np.linspace(range[0], range[1], nbins*nintegration)
    l = []
    for z in x:
        l.append(likef(z, **params))
    L = np.exp(np.array(l))

    figuredata = {'hist':hist, 'bins':output['edges'][:-1], \
        'like':L.copy(), 'x':x}

    L = L.reshape(nbins, nintegration)
    like = L.mean(1)
    return hist, like, figuredata

def discrete_consistency(randomf, likef, params,nrandom=1000, \
    range=None,plot=None):
    """Check the random generator is consistent with the likelihood for
    discrete distributions.

    :Parameters:
      - `randomf`: function: Random generator.
      - `likef`: function: Log probability.
      - `params`: dict:  Parameters for the distribution.
      - `nbins`: int: Number of bins in histogram.
      - `nrandom`: int: Number of random samples.
      - `range`: (float,float): Range of histogram.

    :Return (hist, like):
      - `hist`: Histogram of random samples.
      - `like`: likelihood of histogram bins.
    """
    samples = randomf(size=nrandom, **params)
    hist = np.bincount(samples)*1./nrandom
    x = np.arange(len(hist))
    l = []
    for z in x:
        l.append(likef(z, **params))
    like = np.exp(np.array(l))
    figuredata = {'hist':hist, 'bins':x, \
        'like':like.copy(), 'x':x, 'discrete':True}
    return hist, like, figuredata


def mv_consistency(random, like, params, nbins=10, nrandom=1000, nintegration=15,\
    range=None, plot=None):
    """Check consistency for multivariate distributions."""
    samples = random(n=nrandom, **params)
    hist, edges = np.histogramdd(samples.T, nbins, range, True)
    z = []
    for s in samples:
        z.append(like(s, **params))
    P = np.exp(np.array(z))

def compare_hist(hist, bins, like, x, figname, discrete=False):
    """Plot and save a figure comparing the histogram with the
    probability.

    :Parameters:
      - `samples`: random variables.
      - `like`: probability values.
      - `bins`: histogram bins.
      - `x`: values at which like is computed.
      - `figname`: Name of figure to save.
    """
    ax = P.subplot(111)
    width = 0.9*(bins[1]-bins[0])
    ax.bar(bins, hist, width)
    P.setp(ax.patches, alpha=.5)
    if discrete:
        ax.plot(x, like, 'k', linestyle='steps')
    else:
        ax.plot(x, like, 'k')
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
        hist, like, figdata = discrete_consistency(rbinomial, flib.binomial, \
            params, nrandom=2000)
        assert_array_almost_equal(hist, like,1)
        if PLOT:
            compare_hist(figname='binomial', **figdata)
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
        params = {'k':2}
        hist, like, figdata = consistency(rchi2, chi2_like, params, range=[0,15])
        if PLOT:
            compare_hist(figname='chi2', **figdata)
        assert_array_almost_equal(hist, like, 1)


class test_dirichlet(NumpyTestCase):
    """Multivariate Dirichlet distribution"""
    def check_random(self):
        theta = np.array([2.,3.])
        r = rdirichlet(theta, 2000)
        s = theta.sum()
        m = r.mean(0)
        cov_ex = np.cov(r.transpose())

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

    def check_vectorization(self):
        theta = np.array([[2.,3.], [2,3]])
        r = rdirichlet(theta)
        a = dirichlet_like(r, theta)
        b = dirichlet_like(r, theta[0])
        assert_equal(a,b)
        
    def normalization_2d(self):
        pass

class test_exponential(NumpyTestCase):
    """Based on gamma."""
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
        a = flib.gamma(array([4,5],dtype=float),array(3,dtype=float),array(2,dtype=float))
        b = flib.gamma(array([4,5],dtype=float), array([3,3],dtype=float),array([2,2],dtype=float))
        assert_equal(a,b)

    def check_normalization(self):
        params={'alpha':3, 'beta':2}
        integral = normalization(flib.gamma, params, array([.01,20]), 200)
        assert_almost_equal(integral, 1, 2)

class test_geometric(NumpyTestCase):
    """Based on gamma."""
    def check_consistency(self):
        params={'p':.6}
        hist, like, figdata = discrete_consistency(rgeometric, geometric_like, params,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='geometric', **figdata)
        assert_array_almost_equal(hist, like,1)

class test_gev(NumpyTestCase):
    def check_consistency(self):
        params = dict(xi=.1, mu=4, sigma=3)
        hist, like, figdata = consistency(rgev, flib.gev, params,\
            nbins=20, nrandom=5000)
        if PLOT:
            compare_hist(figname='gev', **figdata)
        assert_array_almost_equal(hist, like,1)

    def check_scipy(self):
        x = [-2,1,2,3,4]
        scipy_y = log(genextreme.pdf(x, -.3, 4, 3))
        flib_y = []
        for i in x:
            flib_y.append(flib.gev(i, .3, 4, 3))
        assert_array_almost_equal(scipy_y,flib_y,5)

    def check_vectorization(self):
        a = flib.gev([4,5,6], xi=.2,mu=4,sigma=1)
        b = flib.gev([4,5,6], xi=[.2,.2,.2],mu=4,sigma=1)
        c = flib.gev([4,5,6], xi=.2,mu=[4,4,4],sigma=1)
        d = flib.gev([4,5,6], xi=.2,mu=4,sigma=[1,1,1])
        assert_equal(a,b)
        assert_equal(b,c)
        assert_equal(c,d)

class test_half_normal(NumpyTestCase):
    def check_consistency(self):
        params={'tau':.5}
        hist, like, figdata = consistency(rhalf_normal, flib.hnormal, params,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='hnormal', **figdata)
        assert_array_almost_equal(hist, like,1)

    def normalization(self):
        params = {'tau':2.}
        integral = normalization(flib.hnormal, params, [0, 20], 200)
        assert_almost_equal(integral, 1, 3)

    def check_vectorization(self):
        a = flib.hnormal([2,3], .5)
        b = flib.hnormal([2,3], tau=[.5,.5])
        assert_equal(a,b)

class test_hypergeometric(NumpyTestCase):
    def check_consistency(self):
        params=dict(draws=10, success=20, failure=12)
        hist, like, figdata = discrete_consistency(rhypergeometric, \
        hypergeometric_like, params, nrandom=5000)
        if PLOT:
            compare_hist(figname='hypergeometric', **figdata)
        assert_array_almost_equal(hist, like,1)

class test_inverse_gamma(NumpyTestCase):
    def check_consistency(self):
        params=dict(alpha=1.5, beta=.5)
        hist, like, figdata = consistency(rinverse_gamma, flib.igamma, params,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='inverse_gamma', **figdata)
        assert_array_almost_equal(hist, like,1)

    def normalization(self):
        params=dict(alpha=1.5, beta=.5)
        integral = normalization(flib.igamma, params, [0, 10], 200)
        assert_almost_equal(integral, 1, 3)

    def vectorization(self):
        x = [2,3]
        a = flib.igamma(x, alpha=[1.5, 1.5], beta=.5)
        b = flib.igamma(x, alpha=[1.5, 1.5], beta=[.5, .5])
        assert_almost_equal(a,b,6)

class test_lognormal(NumpyTestCase):
    def check_consistency(self):
        params=dict(mu=3, tau = .5)
        hist, like, figdata = consistency(rlognormal, flib.lognormal, params,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='lognormal', **figdata)
        assert_array_almost_equal(hist, like,1)

    def normalization(self):
        params=dict(mu=3, tau = .5)
        integral = normalization(flib.lognormal, params, [0, 20], 200)
        assert_almost_equal(integral, 1, 3)

class test_multinomial(NumpyTestCase):
    def check_random(self):
        p = array([.2,.3,.5])
        n = 10
        r = rmultinomial(n=10, p=p, size=5000)
        rmean = r.mean(0)
        assert_array_almost_equal(rmean, n*p, 1)
        rvar = r.var(0)
        assert_array_almost_equal(rvar, n*p*(1-p),1)

    def check_consistency(self):
        p = array([.2,.3,.5])
        n = 10
        x = rmultinomial(n, p, size=5)
        a = multinomial_like(x,n,p)
        b = log(utils.multinomial(x,n,p).prod())
        assert_almost_equal(a,b,4)

    def check_vectorization(self):
        assert_equal(0,1)

class test_multivariate_hypergeometric(NumpyTestCase):
    pass
        
class test_multivariate_normal(NumpyTestCase):
    def check_random(self):
        mu = [3,4]
        C = [[1, .5],[.5,1]]
        r = rmultivariate_normal(mu, np.linalg.inv(C), 1000)
        assert_array_almost_equal(mu, r.mean(0))
        assert_array_almost_equal(C, cov(r))
    
    def check_likelihood(self):
        mu = [3,4]
        C = [[1, .5],[.5,1]]
        r = rmultivariate_normal(mu, np.linalg.inv(C), 2)
        a = multivariate_normal_like(r, mu, C)
        b = prod([utils.multivariate_normal(x, mu, C) for x in r])
        assert_almost_equal(exp(a), b)
    
class test_normal(NumpyTestCase):
    def check_consistency(self):
        params=dict(mu=3, tau = .5)
        hist, like, figdata = consistency(rnormal, flib.normal, params,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='normal', **figdata)
        assert_array_almost_equal(hist, like,1)

    def check_vectorization(self):
        a = flib.normal([3,4,5], mu=3, tau=.5)
        b = flib.normal([3,4,5], mu=[3,3,3], tau=.5)
        c = flib.normal([3,4,5], mu=[3,3,3], tau=[.5,.5,.5])
        assert_equal(a,b)
        assert_equal(b,c)

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

class test_wishart(NumpyTestCase):
    pass


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
