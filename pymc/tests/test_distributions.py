"""Test the distributions from flib and their random generators to check
they are consistent with one another.
For each distribution:
1. Select stochs (stochs = {'alpha':.., 'beta'=...}).
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
from pymc.distributions import *
import unittest
from numpy.testing import *
from pymc import flib, utils
import numpy as np
from numpy import exp, array, cov, prod, matrix
from numpy.linalg import cholesky
import os, pdb
PLOT=True
if PLOT is True:
    try:
        os.mkdir('figs')
    except:
        pass
try:
    from scipy import integrate, special, factorial, comb
    from scipy.stats import genextreme, exponweib
    from scipy.optimize import fmin
    SP = True
except:
    print 'Some of the tests might not pass because they depend on SciPy functions.'
    SP= False
try:
    import pylab as P
except:
    print 'Plotting disabled'
    PLOT=False
SP = False

# Some python densities for comparison
def cauchy(x, x0, gamma):
    return 1/pi * gamma/((x-x0)**2 + gamma**2)

def gamma(x, alpha, beta):
    return x**(alpha-1) * exp(-x/beta)/(special.gamma(alpha) * beta**alpha)

def multinomial_beta(alpha):
    nom = (special.gamma(alpha)).prod(0)
    den = special.gamma(alpha.sum(0))
    return nom/den

def dirichlet(x, theta):
    """Dirichlet multivariate probability density.

    :Stochastics:
      x : (n,k) array
        Input data
      theta : (n,k) or (1,k) array
        Distribution stoch
    """
    x = np.atleast_2d(x)
    theta = np.atleast_2d(theta)
    f = (x**(theta-1)).prod(0)
    return f/multinomial_beta(theta)

def geometric(x, p):
    return p*(1.-p)**(x-1)

def hypergeometric(x, n, m, N):
    """
    x : number of successes drawn
    n : number of draws
    m : number of successes in total
    N : nuccesses + failures in total.
    """
    return comb(N-m, x) * comb(m, n-x) / comb(N,n)

def mv_hypergeometric(x,m):
    """
    x : number of draws for each category.
    m : size of each category.
    """
    x = np.asarray(x)
    m = np.asarray(m)
    return log(comb(m,x).prod()/comb(m.sum(), x.sum()))

def multinomial(x,n,p):
    x = np.atleast_2d(x)
    return factorial(n)/factorial(x).prod(1)*(p**x).prod(1)

def mv_normal(x, mu, C):
    N = len(x)
    x = np.asmatrix(x)
    mu = np.asmatrix(mu)
    C = np.asmatrix(C)

    I = (N/2.)*log(2.*pi) + .5*log(np.linalg.det(C))
    z = (x-mu)

    return -(I +.5 * z * np.linalg.inv(C) * z.T).A[0][0]

def multivariate_lognormal(x, mu, C):
    N = len(x)
    x = asmatrix(x)
    mu = asmatrix(mu)
    C = asmatrix(C)
    
    I = (2*pi)**(N/2.) * sqrt(det(C))
    z = (np.log(x)-mu)
    return (1./I * exp(-.5 * z * inv(C) * z.T)).A[0][0]



def consistency(randomf, likef, stochs, nbins=10, nrandom=1000, nintegration=15,\
    range=None, plot=None):
    """Check the random generator is consistent with the likelihood.

    :Stochastics:
      - `random`: function: Random generator.
      - `like`: function: Log probability.
      - `stochs`: dict:  Stochastics for the distribution.
      - `nbins`: int: Number of bins in histogram.
      - `nrandom`: int: Number of random samples.
      - `nintegration`: int: Number of divisions in each bin.
      - `range`: (float,float): Range of histogram.

    :Return (hist, like):
      - `hist`: Histogram of random samples from random(**stochs).
      - `like`: integrated likelihood over histogram bins.
    """
    # Samples values and compute histogram.
    samples = randomf(size=nrandom, **stochs)

    hist, output = utils.histogram(samples, range=range, bins=nbins, normed=True)

    # Compute likelihood along x axis.
    if range is None:
        range = samples.min(), samples.max()
    x = np.linspace(range[0], range[1], nbins*nintegration)
    l = []
    for z in x:
        l.append(likef(z, **stochs))
    L = np.exp(np.array(l))

    figuredata = {'hist':hist, 'bins':output['edges'][:-1], \
        'like':L.copy(), 'x':x}

    L = L.reshape(nbins, nintegration)
    like = L.mean(1)
    return hist, like, figuredata

def discrete_consistency(randomf, likef, stochs,nrandom=1000, \
    range=None,plot=None):
    """Check the random generator is consistent with the likelihood for
    discrete distributions.

    :Stochastics:
      - `randomf`: function: Random generator.
      - `likef`: function: Log probability.
      - `stochs`: dict:  Stochastics for the distribution.
      - `nbins`: int: Number of bins in histogram.
      - `nrandom`: int: Number of random samples.
      - `range`: (float,float): Range of histogram.

    :Return (hist, like):
      - `hist`: Histogram of random samples.
      - `like`: likelihood of histogram bins.
    """
    samples = randomf(size=nrandom, **stochs)
    hist = np.bincount(samples)*1./nrandom
    x = np.arange(len(hist))
    l = []
    for z in x:
        l.append(likef(z, **stochs))
    like = np.exp(np.array(l))
    figuredata = {'hist':hist, 'bins':x, \
        'like':like.copy(), 'x':x, 'discrete':True}
    return hist, like, figuredata


def mv_consistency(random, like, stochs, nbins=10, nrandom=1000, nintegration=15,\
    range=None, plot=None):
    """Check consistency for multivariate distributions."""
    samples = random(n=nrandom, **stochs)
    hist, edges = np.histogramdd(samples.T, nbins, range, True)
    z = []
    for s in samples:
        z.append(like(s, **stochs))
    P = np.exp(np.array(z))

def compare_hist(hist, bins, like, x, figname, discrete=False):
    """Plot and save a figure comparing the histogram with the
    probability.

    :Stochastics:
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

def normalization(like, stochs, domain, N=100):
    """Integrate the distribution over domain.

    :Stochastics:
      - `like`: log probability density.
      - `stochs`: {}: distribution stochs.
      - `domain`: domain of integration.
      - `N`:  Number of samples for trapezoidal integration.

    Note:
      The integration is performed using scipy.integrate.quadg if available.
      Otherwise, a trapezoidal integration is done.
    """
    f = lambda x: exp(like(x, **stochs))
    if SP:
        out = integrate.quad(f, domain[0], domain[1])
        return out[0]
    else:
        y = []
        x = np.linspace(domain[0], domain[1], N)
        for i in x:
            y.append(f(i))
        return np.trapz(y,x)

class test_arnormal(NumpyTestCase):
    def like(self, stochs, r):
        a = stochs[2:]
        tau = stochs[1]
        rho = stochs[0]
        like = np.array([arnormal_like(x, a, tau, rho) for x in r])
        return -like.sum()
            
    def check_random(self):
        a = (1,2)
        tau = 10
        rho = 0
        r = rarnormal(a, tau, rho, size=1000) 
        assert_array_almost_equal(np.median(r), [1,2],1)
        
        rho =.8
        sigma = .1
        r = rarnormal(1, tau, rho, size=1000)
        corr = utils.autocorr(np.log(r))
        assert_almost_equal(corr, rho, 1)
        assert_almost_equal(r.std(), tau/sqrt(1-rho**2),1)
    
    def check_consistency(self):
    
        # 1D case
        a = 1
        rho =.8
        tau = 10
        r = rarnormal(a, tau, rho, size=1000)
        opt = fmin(self.like, (.8, .4, .9), args=([r],), disp=0)
        assert_array_almost_equal(opt, [rho, tau, a], 1)

        # 2D case
        a = (1,2)
        tau = 10
        rho = .7
        r = rarnormal(a, tau, rho, size=2000) 
        opt = fmin(self.like, (.75, .15, 1.1, 2.1), xtol=.05, args=(r,), disp=0)
        assert_array_almost_equal(opt, (rho, tau)+a, 1)
    

class test_bernoulli(NumpyTestCase):
    def check_consistency(self):
        N = 5000
        stochs = {'p':.6}
        samples = []
        for i in range(N):
            samples.append(rbernoulli(**stochs))
        H = np.bincount(samples)*1./N
        l0 = exp(flib.bernoulli(0, **stochs))
        l1 = exp(flib.bernoulli(1, **stochs))
        assert_array_almost_equal(H, [l0,l1], 2)
        # Check normalization
        assert_almost_equal(l0+l1, 1, 4)

    def test_calling(self):
        a = flib.bernoulli([0,1,1,0], .4)
        b = flib.bernoulli([0,1,1,0], [.4, .4, .4, .4])
        assert_array_equal(b,a)

class test_beta(NumpyTestCase):
    def check_consistency(self):
        stochs ={'alpha':3, 'beta':5}
        hist, like, figdata = consistency(rbeta, flib.beta_like, stochs, nrandom=5000, range=[0,1])
        assert_array_almost_equal((hist-like)/sqrt(like),0,1)
        if PLOT:
            compare_hist(figname='beta', **figdata)

    def test_calling(self):
        a = flib.beta_like([.3,.4,.5], 2,3)
        b = flib.beta_like([.3,.4,.5], [2,2,2],[3,3,3])
        assert_array_equal(a,b)

    def check_normalization(self):
        stochs ={'alpha':3, 'beta':5}
        integral = normalization(flib.beta_like, stochs, [0,1], 200)
        assert_almost_equal(integral, 1, 3)

class test_binomial(NumpyTestCase):
    def check_consistency(self):
        stochs={'n':7, 'p':.7}
        hist, like, figdata = discrete_consistency(rbinomial, flib.binomial, \
            stochs, nrandom=2000)
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
        stochs={'alpha':0, 'beta':.5}
        hist, like, figdata = consistency(rcauchy, flib.cauchy, stochs, \
        nrandom=5000, range=[-10,10], nbins=21)
        if PLOT:
            compare_hist(figname='cauchy', **figdata)
        assert_array_almost_equal(hist, like,1)


    def test_calling(self):
        a = flib.cauchy([3,4], 2, 6)
        b = flib.cauchy([3,4], [2,2], [6,6])
        assert_equal(a,b)

    def check_normalization(self):
        stochs={'alpha':0, 'beta':.5}
        integral = normalization(flib.cauchy, stochs, [-100,100], 600)
        assert_almost_equal(integral, 1, 2)

class test_chi2(NumpyTestCase):
    """Based on flib.gamma, so no need to make the calling check and
    normalization check."""
    def check_consistency(self):
        stochs = {'nu':2}
        hist, like, figdata = consistency(rchi2, chi2_like, stochs, range=[0,15])
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
        f = dirichlet(x, theta)
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
        stochs={'beta':4}
        hist, like, figdata = consistency(rexponential, exponential_like, 
            stochs, nrandom=5000)
        if PLOT:
            compare_hist(figname='exponential', **figdata)
        assert_array_almost_equal(hist, like,1)

class test_exponweib(NumpyTestCase):
    def check_consistency(self):
        stochs = {'alpha':2, 'k':2, 'loc':1, 'scale':3}
        hist,like,figdata=consistency(rexponweib, exponweib_like, 
            stochs, nrandom=5000)
        if PLOT:
            compare_hist(figname='exponweib', **figdata)
        assert_array_almost_equal(hist, like, 1)

    def check_random(self):
        r = rexponweib(2, 1, 4, 5, size=1000)
        r.mean(), r.var()
        # scipy.exponweib.stats is buggy. may 17, 2007

    def check_with_scipy(self):
        stochs = {'alpha':2, 'k':.3, 'loc':1, 'scale':3}
        r = rexponweib(size=10, **stochs)
        a = exponweib.pdf(r, 2,.3, 1, 3)
        b = exponweib_like(r, **stochs)
        assert_almost_equal(log(a).sum(), b, 6)

class test_gamma(NumpyTestCase):
    def check_consistency(self):
        stochs={'alpha':3, 'beta':2}
        hist, like, figdata = consistency(rgamma, flib.gamma, stochs,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='gamma', **figdata)
        assert_array_almost_equal(hist, like,1)

    def test_calling(self):
        a = flib.gamma(array([4,5],dtype=float),array(3,dtype=float),array(2,dtype=float))
        b = flib.gamma(array([4,5],dtype=float), array([3,3],dtype=float),array([2,2],dtype=float))
        assert_equal(a,b)

    def check_normalization(self):
        stochs={'alpha':3, 'beta':2}
        integral = normalization(flib.gamma, stochs, array([.01,20]), 200)
        assert_almost_equal(integral, 1, 2)

class test_geometric(NumpyTestCase):
    """Based on gamma."""
    def check_consistency(self):
        stochs={'p':.6}
        hist, like, figdata = discrete_consistency(rgeometric, geometric_like, stochs,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='geometric', **figdata)
        assert_array_almost_equal(hist, like,1)

class test_gev(NumpyTestCase):
    def check_consistency(self):
        stochs = dict(xi=.1, mu=4, sigma=3)
        hist, like, figdata = consistency(rgev, flib.gev, stochs,\
            nbins=20, nrandom=5000)
        if PLOT:
            compare_hist(figname='gev', **figdata)
        assert_array_almost_equal(hist, like,1)

    def check_scipy(self):
        x = [1,2,3,4]
        scipy_y = log(genextreme.pdf(x, -.3, 4, 2))
        flib_y = []
        for i in x:
            flib_y.append(flib.gev(i, .3, 4, 2))
        assert_array_almost_equal(scipy_y,flib_y,5)

    def check_limit(self):
        x = [1,2,3]
        a = flib.gev(x, 0.00001, 0, 1)
        b = flib.gev(x, 0, 0, 1)
        assert_almost_equal(a,b,4)

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
        stochs={'tau':.5}
        hist, like, figdata = consistency(rhalf_normal, flib.hnormal, stochs,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='hnormal', **figdata)
        assert_array_almost_equal(hist, like,1)

    def normalization(self):
        stochs = {'tau':2.}
        integral = normalization(flib.hnormal, stochs, [0, 20], 200)
        assert_almost_equal(integral, 1, 3)

    def check_vectorization(self):
        a = flib.hnormal([2,3], .5)
        b = flib.hnormal([2,3], tau=[.5,.5])
        assert_equal(a,b)

class test_hypergeometric(NumpyTestCase):
    def check_consistency(self):
        stochs=dict(n=10, m=12, N=20)
        hist, like, figdata = discrete_consistency(rhypergeometric, \
        hypergeometric_like, stochs, nrandom=5000)
        if PLOT:
            compare_hist(figname='hypergeometric', **figdata)
        assert_array_almost_equal(hist, like,1)

class test_inverse_gamma(NumpyTestCase):
    def check_consistency(self):
        stochs=dict(alpha=1.5, beta=.5)
        hist, like, figdata = consistency(rinverse_gamma, flib.igamma, stochs,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='inverse_gamma', **figdata)
        assert_array_almost_equal(hist, like,1)

    def normalization(self):
        stochs=dict(alpha=1.5, beta=.5)
        integral = normalization(flib.igamma, stochs, [0, 10], 200)
        assert_almost_equal(integral, 1, 3)

    def vectorization(self):
        x = [2,3]
        a = flib.igamma(x, alpha=[1.5, 1.5], beta=.5)
        b = flib.igamma(x, alpha=[1.5, 1.5], beta=[.5, .5])
        assert_almost_equal(a,b,6)

class test_lognormal(NumpyTestCase):
    def check_consistency(self):
        stochs=dict(mu=3, tau = .5)
        hist, like, figdata = consistency(rlognormal, flib.lognormal, stochs,\
            nrandom=5000)
        if PLOT:
            compare_hist(figname='lognormal', **figdata)
        assert_array_almost_equal(hist, like,1)

    def normalization(self):
        stochs=dict(mu=3, tau = .5)
        integral = normalization(flib.lognormal, stochs, [0, 20], 200)
        assert_almost_equal(integral, 1, 3)
        
    def check_vectorization(self):
        r = rlognormal(3, .5, 2)
        a = lognormal_like(r, 3, .5)
        b = lognormal_like(r, [3,3], [.5,.5])
        assert_array_equal(a,b)

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
        a = multinomial_like(x[:,:-1],n,p[:-1])
        b = log(multinomial(x,n,p).prod())
        assert_almost_equal(a,b,4)

    def check_vectorization(self):
        p = array([[.2,.3,.5], [.2,.3,.5]])
        r = rmultinomial(10, p=p[0], size=2)
        a = multinomial_like(r[:,:-1],10,p[0,:-1])
        b = multinomial_like(r[:,:-1],[10,10],p[:,:-1])
        assert_equal(a,b)

class test_multivariate_hypergeometric(NumpyTestCase):
    def check_random(self):
        m = [10,15]
        N = 200
        n = 6
        r = rmultivariate_hypergeometric(n, m, N)
        assert_array_almost_equal(r.mean(0), multivariate_hypergeometric_expval(n,m),1)
        
    def check_likelihood(self):
        m = [10,15]
        x = [3,4]
        a = multivariate_hypergeometric_like(x, m)
        b = mv_hypergeometric(x, m)
        assert_almost_equal(a,b,4)


class test_mv_normal(NumpyTestCase):
    
    def check_random(self):
        mu = array([3,4])
        C = matrix([[1, .5],[.5,1]])

        r = rmv_normal(mu, np.linalg.inv(C), 1000)
        rC = rmv_normal_cov(mu,C,1000)
        rchol = rmv_normal_chol(mu,cholesky(C),1000)
        
        assert_array_almost_equal(mu, r.mean(0), 1)
        assert_array_almost_equal(C, np.cov(r.T), 1)
        
        assert_array_almost_equal(mu, rC.mean(0), 1)
        assert_array_almost_equal(C, np.cov(rC.T), 1)
        
        assert_array_almost_equal(mu, rchol.mean(0), 1)
        assert_array_almost_equal(C, np.cov(rchol.T), 1)
            
    def check_likelihood(self):
        mu = array([3,4])
        C = matrix([[1, .5],[.5,1]])
        

        tau = np.linalg.inv(C)
        r = rmv_normal(mu, tau, 2)

        # mv_normal_like is expecting tau as its last argument
        a = sum([mv_normal_like(x, mu, tau) for x in r])
        b = sum([mv_normal_cov_like(x, mu, C) for x in r])

        # mv_normal_like_chol is expecting a Cholesky factor as the last argument.
        c = sum([mv_normal_chol_like(x,mu,cholesky(C)) for x in r])
        d = sum([mv_normal(x, mu, C) for x in r])

        assert_almost_equal(a, d,6)
        assert_almost_equal(a,b,6)
        assert_almost_equal(b,c,6)

    
class test_normal(NumpyTestCase):
    def check_consistency(self):
        stochs=dict(mu=3, tau = .5)
        hist, like, figdata = consistency(rnormal, flib.normal, stochs,\
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
        stochs = {'mu':2.}
        hist,like, figdata = discrete_consistency(rpoisson, flib.poisson, stochs, nrandom=5000, range=[0,10])
        if PLOT:
            compare_hist(figname='poisson', **figdata)
        assert_array_almost_equal(hist, like,1)

    def normalization(self):
        stochs = {'mu':2.}
        integral = normalization(flib.poisson, stochs, [0.1, 20], 200)
        assert_almost_equal(integral, 1, 2)

    def test_calling(self):
        a = flib.poisson([1,2,3], 2)
        b = flib.poisson([1,2,3], [2,2,2])
        assert_equal(a,b)
        
class test_truncnorm(NumpyTestCase):
    def check_consistency(self):
        stochs = dict(mu=1, sigma=1, a=0, b=5)
        hist,like, figdata = consistency(rtruncnorm, truncnorm_like, stochs, nrandom=5000)
        if PLOT:
            compare_hist(figname='truncnorm', **figdata)
        assert_array_almost_equal(hist, like,1)
        
    def normalization(self):
        stochs = dict(mu=1, sigma=1, a=0, b=2)
        integral = normalization(truncnorm_like, stochs, [-1, 3], 200)
        assert_almost_equal(integral, 1, 2)
        
    def test_calling(self):
        a = truncnorm_like([0,1,2], 2, 1, 0, 2)
        b = truncnorm_like([0,1,2], [2,2,2], 1, 0, 2)
        c = truncnorm_like([0,1,2], [2,2,2], [1,1,1], 0,2)
        #d = truncnorm_like([0,1,2], [2,2,2], [1,1,1], [0,0,0], 2)
        #e = truncnorm_like([0,1,2], [2,2,2], [1,1,1], 0, [2,2,2])
        assert_equal(a,b)
        assert_equal(a,b)
        assert_equal(a,c)
        #assert_equal(a,d)
        #assert_equal(a,e)
    
class test_weibull(NumpyTestCase):
    def check_consistency(self):
        stochs = {'alpha': 2., 'beta': 3.}
        hist,like, figdata = consistency(rweibull, flib.weibull, stochs, nrandom=5000, range=[0,10])
        if PLOT:
            compare_hist(figname='weibull', **figdata)
        assert_array_almost_equal(hist, like,1)

    def check_norm(self):
        stochs = {'alpha': 2., 'beta': 3.}
        integral = normalization(flib.weibull, stochs, [0,10], N=200)
        assert_almost_equal(integral, 1, 2)

    def test_calling(self):
        a = flib.weibull([1,2], 2,3)
        b = flib.weibull([1,2], [2,2], [3,3])
        assert_equal(a,b)

class test_wishart(NumpyTestCase):
    """How do we know it's really working ?"""
    def check(self):
        S = [[2,.4],[.4,3]]
        T = np.linalg.inv(S)
        n=2
        r = rwishart(n, T)
        wishart_like(r, n, T)
        assert_equal(0,1)


"""
Hyperg is stochetrized differently in flib than is hypergeometric in numpy.random.
numpy.random: good, bad, sample
flib: red (=bad), d (=sample), total (=good+bad)

What to do about this?
Either change the argument names in flib or in the distributions wrapper.
-D

class test_hyperg(NumpyTestCase):
    def __init__(self):
        from np.random import poisson
        stochs = {'ngood':2, 'nbad':5, 'nsample':3}
        hist,like, figdata = consistency(poisson, flib.poisson, stochs, nrandom=5000, range=[0,10])
"""

if __name__ == '__main__':
    NumpyTest().run()
