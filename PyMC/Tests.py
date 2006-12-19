"""Test of PyMC features, functions and calling conventions."""
 
import sys, unittest, PyMC
from PyMC import MetropolisHastings
from numpy import *

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

if __name__=='__main__':
    unittest.main()
    

