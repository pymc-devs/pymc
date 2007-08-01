"""
author: Chris Fonnesbeck
date: 2007-07-31
encoding: utf-8
"""

from PyMC import MetropolisHastings
import unittest

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
        


if __name__ == '__main__':
    unittest.main()

