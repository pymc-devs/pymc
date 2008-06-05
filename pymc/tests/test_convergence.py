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
S.sample(5000, 2500)
a = S.a.trace()
b = S.b.trace()

class test_geweke(NumpyTestCase):
    def check_simple(self):
        scores = pymc.geweke(a, intervals=5)
        assert_equal(len(scores), 5)
        
        # If the model has converged, the scores should lie
        # between -1 and 1.
        assert_array_equal(np.abs(np.array(scores)[:,1]) < 1, True)
        
class test_raftery_lewis(NumpyTestCase):
    def check_simple(self):
        pymc.raftery_lewis(a, 0.5, .05)


if __name__ == "__main__":
    NumpyTest().run()
