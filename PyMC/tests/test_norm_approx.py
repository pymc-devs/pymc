from PyMC import NormalApproximation, msqrt
from PyMC.examples import gelman_bioassay
from pylab import *
from numpy import *
from numpy.testing import * 
from numpy.linalg import cholesky

model = gelman_bioassay
N = NormalApproximation(model)

class test_norm_approx(NumpyTestCase):
    def check_fmin(self):
        N = NormalApproximation(model)
        N.fit('fmin')
    def check_fmin_l_bfgs_b(self):
        N = NormalApproximation(model)
        N.fit('fmin_l_bfgs_b')
    def check_fmin_ncg(self):
        N = NormalApproximation(model)
        N.fit('fmin_ncg')
    def check_fmin_cg(self):
        N = NormalApproximation(model)
        N.fit('fmin_cg')
    def check_fmin_powell(self):
        N = NormalApproximation(model)
        N.fit('fmin_powell')
    def check_sig(self):
        N = NormalApproximation(model)
        N.fit('fmin')
        assert((abs(N._sig * N._sig.T - N._C) < 1.0e-14).all())        
    def check_draws(self):
        N = NormalApproximation(model)
        N.fit('fmin')
        N.sample(1000)

        plot(N.alpha.trace(),N.beta.trace(),'k.')
        xlabel(r'$\alpha$')
        ylabel(r'$\beta$')
        
if __name__=='__main__':
    NumpyTest().run()