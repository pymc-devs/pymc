###
#
# Test of convergence diagnostics
#
###


from numpy.testing import assert_equal, assert_array_equal, NumpyTestCase, NumpyTest
import numpy as np
import pymc
import pymc.examples.weibull_fit as model

S = pymc.MCMC(model, 'ram')
S.sample(10000, 5000)
a = S.a.trace()
b = S.b.trace()

class test_geweke(NumpyTestCase):
    def check_simple(self):
        scores = pymc.geweke(a, intervals=20)
        assert_equal(len(scores), 20)
        
        # If the model has converged, 95% the scores should lie
        # within 2 standard deviations of zero, under standard normal model
        assert(sum(np.abs(np.array(scores)[:,1]) > 1.96) < 2)
        
class test_raftery_lewis(NumpyTestCase):
    def check_simple(self):
        nmin, kthin, nburn, nprec, kmind = pymc.raftery_lewis(a, 0.5, .05, verbose=1)
        
        # nmin should approximately be the same as nprec/kmind
        assert(0.8 < (float(nprec)/kmind) / nmin < 1.2)

if __name__ == "__main__":
    NumpyTest().testall(level=10, verbosity=10)
