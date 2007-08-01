"""
author: Chris Fonnesbeck
date: 2007-07-31
encoding: utf-8
"""

from PyMC import *
import unittest

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
        
        self.failIf(self.sampler.p < lo or self.sampler.p > hi)
        
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