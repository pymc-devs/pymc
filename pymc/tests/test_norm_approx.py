from pymc import NormApprox
from pymc.examples import gelman_bioassay
try:
    from pylab import *
except:
    pass
    
from numpy import *
from numpy.testing import * 
from numpy.linalg import cholesky
import nose
PLOT=False

model = gelman_bioassay

class test_norm_approx(TestCase):
    @classmethod
    def setUpClass(self):
        try:
            import scipy
        except:
            raise nose.SkipTest,"SciPy not installed."
    def test_fmin(self):
        N = NormApprox(model)
        N.fit('fmin')
    def test_fmin_l_bfgs_b(self):
        N = NormApprox(model)
        N.fit('fmin_l_bfgs_b')
    def test_fmin_ncg(self):
        N = NormApprox(model)
        N.fit('fmin_ncg')
    def test_fmin_cg(self):
        N = NormApprox(model)
        N.fit('fmin_cg')
    def test_fmin_powell(self):
        N = NormApprox(model)
        N.fit('fmin_powell')
    def test_sig(self):
        N = NormApprox(model)
        N.fit('fmin')
        assert((abs(N._sig * N._sig.T - N._C) < 1.0e-14).all())        
    def test_draws(self):
        N = NormApprox(model)
        N.fit('fmin')
        N.sample(1000)
        if PLOT:
            plot(N.alpha.trace(),N.beta.trace(),'k.')
            xlabel(r'$\alpha$')
            ylabel(r'$\beta$')
        
if __name__=='__main__':
    import unittest
    unittest.main()
