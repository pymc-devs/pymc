"""Test of PyMC features, functions and calling conventions."""
 
import sys, unittest, PyMC
from PyMC import *

class CorrelatedSampler(MetropolisHastings):
    """Test case using multivariate jump distribution.
    For now, the test only checks that different calling conventions work."""
    
    def __init__(self, case):
        MetropolisHastings.__init__(self)
        
        # Create synthetic data
        N = 100
        synth_mean = [5,10]
        synth_var = array([1,4])
        rho = .9
        synth_cov = sqrt(outer(synth_var, synth_var)) * [[1,rho], [rho, 1]]
        self.r1, self.r2 = random.multivariate_normal(synth_mean, synth_cov, size=N).T
        self.case = str(case)
        self.parameter('v%d'%case, init_val = [.8, 3.5])
    
        if case == 0:
            self.parameter('m%d'%case, init_val=[4.,11.])
        elif case ==1:
            self.parameter('m%d'%case, init_val=[4,11], dist='multivariate_normal', scale=[1,2])
        elif case == 2:
            self.parameter('m%d'%case, init_val=[4,11], dist='multivariate_normal',scale=[[.9,0.],[0., 3.5]])
        elif case == 3:
            self.parameter('m%d'%case, init_val=array([4,11]), dist='multivariate_normal', scale=array([[.9,0.],[0., 3.5]]))
        else: 
            raise 'No such case %d.' % case

    def calculate_likelihood(self):
        like = 0
        m = getattr(self, 'm'+self.case)
        v = getattr(self, 'v'+self.case)
        like += self.normal_like(self.r1, m[0], 1./v[0])
        like += self.normal_like(self.r2, m[1], 1./v[1])
        return like


class MultiParametersSampler(MetropolisHastings):
    """Test case using multivariate jump distribution.
    For now, the test only checks that different calling conventions work."""
    
    def __init__(self):
        MetropolisHastings.__init__(self)
    
        self.parameter('A', init_val=1)
        self.parameter('B', init_val=[2,2])
        self.parameter('C', init_val = [[1,2],[3,4]])
        self.parameter('D', init_val = 1, scale = 3)
        self.parameter('E', init_val = [2,3], scale=4)
        self.parameter('F', init_val = [2,3], scale = [3,4])
        self.parameter('G', init_val = [[1,2],[3,4]], scale = [[1,1],[2,2]])
    
    def calculate_likelihood(self):
        return self.normal_like(self.A , 2., 1.)

class MCMCTest(unittest.TestCase):
    def testCorrelatedSampler(self):
        """Run correlated sampler test."""
        iterations = 200
        print 'Running correlated sampler test case ...'
        
        self.sampler = {}
        print 'Class initialization ...'
        for case in [0,1,2,3]:
            print 'Case:', case
            self.sampler[case] = CorrelatedSampler(case)
        print '\nSampling ...'
        for case in [0,1,2,3]:
          self.failUnless(self.sampler[case].sample(iterations,100, verbose=True, plot=False))
            
        for case in [0,1,2,3]:
            self.sampler[case].convergence()        
            # Goodness of fit
            x, n = self.sampler[case].goodness(iterations/10)['overall']
            self.failIf(x/n < 0.05 or x/n > 0.95)


    def testMultiParametersSampler(self):
        """Run MultiParameter sampler test."""
        print 'Running MultiParametersSampler test case ...'
        self.sampler = MultiParametersSampler()
        self.failUnless(self.sampler.sample(1000,100, verbose=False, plot=False))
        
class DisasterSampler(MetropolisHastings):
    """
    Test example based on annual coal mining disasters in the UK. Occurrences
    of disasters in the time series is thought to be derived from a
    Poisson process with a large rate parameter in the early part of
    the time series, and from one with a smaller rate in the later part.
    We are interested in locating the switchpoint in the series using
    MCMC.
    """
    
    def init(self):
        """Class initialization"""
        
        """
        try:
            MetropolisHastings.__init__(self, db_backend='sqlite')
        except Exception:
            MetropolisHastings.__init__(self, db_backend='ram')
        """
        
        # Sample changepoint data (Coal mining disasters per year)
        self.data = (4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
            2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
            3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1)
        
        # Switchpoint is specified as a parameter to be estimated
        self.parameter(name='k', init_val=50, discrete=True)
        
        # Rate parameters of poisson distributions
        self.parameter(name='theta', init_val=array([1.0, 1.0]))
    
    def model(self):
        """Joint log-posterior"""
        
        # Obtain current values of parameters as local variables
        theta0, theta1 = self.theta
        k = self.k
        
        # Constrain k with prior
        self.uniform_prior(k, 1, len(self.data)-2)

        # Joint likelihood of parameters based on 2 assumed Poisson densities
        self.poisson_like(self.data[:k], theta0, name='early_mean')

        self.poisson_like(self.data[k:], theta1, name='late_mean')


class MCMCTest(unittest.TestCase):
    
    def testCoalMiningDisasters(self):
        """Run coal mining disasters example sampler"""
        
        print 'Running coal mining disasters test case ...'
        
        # Create an instance of the sampler
        self.sampler = DisasterSampler()
        
        # Specify the nimber of iterations to execute
        iterations = 10000
        thin = 2
        burn = 5000
        chains = 2
        
        # Run MCMC simulation
        for i in range(chains):
            
            self.failUnless(self.sampler.sample(iterations, burn=burn, thin=thin, plot=True, color='r'))
            
            # Run convergence diagnostics
            self.sampler.convergence(burn=burn, thin=thin)
            
            # Plot autocorrelation
            self.sampler.autocorrelation(burn=burn, thin=thin)
        
        # Goodness of fit
        x, n = self.sampler.goodness(iterations/10, burn=burn, thin=thin)['overall']
        self.failIf(x/n < 0.05 or x/n > 0.95)
        
class LikelihoodTest(unittest.TestCase):
    
    def setUp(self):
        """Initialize sampler for testing"""
        
        self.sampler = MetropolisHastings()
        
    def testBinomial(self):
        """Test binomial likelihood"""
        
        print "Testing binomial likelihood"
        
        # 1000 samples from binomial with N=100, p=0.3
        data = rbinomial(100, 0.3, 100)
        
        self.sampler.parameter('p', init_val=0.5)
        
        self.sampler.model = lambda: self.sampler.binomial_like(data, 100, self.sampler.p)
        
        results = self.sampler.sample(1000, burn=500, plot=False, verbose=False)
        
        lo, hi = results['p']['95%s HPD interval' % '%']
        
        self.failIf(p < lo or p > hi)
        
    def testCategorical(self):
        """Test categorical likelihood"""
        
        print "Testing categorical likelihood"
        
        # Sample from categorical distribution
        probs = [0.5, 0.2, 0.2, 0.1]
        minval = 0
        step = 1
        data = rcategorical(probs, minval, step, n=1000)
        
        self.sampler.parameter('p', init_val=array([0.25]*4))
        
        self.sampler.model = lambda: self.sampler.categorical_like(data, self.sampler.p)
        
        self.sampler.sample(500, burn=200, plot=False)
        
    def testBeta(self):
        """Test beta likelihood"""
        
        print "Testing beta likelihood"
        
        # Sample from beta distribution
        alpha = 2
        beta = 10
        data = rbeta(alpha, beta, 100)
        
        self.sampler.parameter('alpha', init_val=1.0)
        self.sampler.parameter('beta', init_val=1.0)
        
        self.sampler.model = lambda: self.sampler.beta_like(data, self.sampler.alpha, self.sampler.beta)
        
        results = self.sampler.sample(1000, burn=500, plot=False, verbose=False)
        
        lo, hi = results['alpha']['95%s HPD interval' % '%']
        print 'alpha 95%s BCI = [%s, %s], True value = %s\n' % ('%', lo, hi, alpha)
        self.failIf(alpha < lo or alpha > hi)
        
        lo, hi = results['beta']['95%s HPD interval' % '%']
        print 'beta 95%s BCI = [%s, %s], True value = %s\n' % ('%', lo, hi, beta)
        self.failIf(beta < lo or beta > hi)
        
    def testDirichlet(self):
        """Test dirichlet likelihood"""
        
        print "Testing dirichlet likelihood"
        
        # Sample from dirichlet distribution
        theta = [0.4,0.3,0.2,0.1]
        data = rdirichlet(theta, 100)
        
        self.sampler.parameter('theta', init_val=array([0.25]*4))
        
        self.sampler.model = lambda: self.sampler.dirichlet_like(data, self.sampler.theta)
        
        results = self.sampler.sample(1000, burn=500, plot=False, verbose=False)
        
        lo, hi = results['theta']['95%s HPD interval' % '%']
        for t, l, h in zip(theta, lo, hi):
            print 'theta 95%s BCI = [%s, %s], True value = %s\n' % ('%', l, h, t)
            self.failIf(t < lo or t > hi)
        
    def testNegativeBinomial1(self):
        """Test negative binomial likelihood"""
        
        print "Testing negative binomial likelihood"
        
        # Sample from negative binomial distribution
        r = 10
        p = 0.25
        data = rnegbin(r, p, 100)
        
        self.sampler.parameter('r', init_val=5)
        self.sampler.parameter('p', init_val=0.5)
        
        self.sampler.model = lambda: self.sampler.negative_binomial_like(data, self.sampler.r, self.sampler.p)
        
        results = self.sampler.sample(1000, burn=500, plot=False, verbose=False)
        
        lo, hi = results['r']['95%s HPD interval' % '%']
        print 'r 95%s BCI = [%s, %s], True value = %s\n' % ('%', lo, hi, r)
        self.failIf(r < lo or r > hi)
        
        lo, hi = results['p']['95%s HPD interval' % '%']
        print 'p 95%s BCI = [%s, %s], True value = %s\n' % ('%', lo, hi, p)
        self.failIf(p < lo or p > hi)

if __name__=='__main__':
    # Run unit tests
    unittest.main()

