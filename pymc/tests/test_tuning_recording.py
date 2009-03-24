import nose
import pymc as pm
from numpy import mean, var
from numpy.testing import *
from pylab import plot

class test_tuning_recording(TestCase):

    def test_Metropolis(self):
        X = pm.Normal('X',0,1)
        @pm.potential
        def Y(X=X):
            return 0
        M = pm.MCMC([X])
        M.use_step_method(pm.Metropolis, X, tally=True)

        M.isample(100000, 0, 10, tune_throughout=True, tune_interval=1000)

        tasf = M.trace('Metropolis_X_adaptive_scale_factor')[:]

        plot(tasf)

    def test_AM(self):
        X = pm.Normal('X',0,1,size=3)
        @pm.potential
        def Y(X=X):
            return 0
        M = pm.MCMC([X])
        M.use_step_method(pm.AdaptiveMetropolis, X, tally=True)
        M.isample(100000, 0, 10)
        
        tc = M.trace('AdaptiveMetropolis_X_C')
        plot(tc[:][:,0,0])
    
if __name__ == '__main__':
    nose.runmodule()