###
#
# Test of convergence diagnostics
#
###


from numpy.testing import *
import PyMC2
import PyMC2.examples.maximum_rainfall as mr

S = PyMC2.Sampler(mr.Gumbelfit, 'ram')
S.sample(1000)
sigma = S.sigma.trace()

class test_geweke(NumpyTestCase):
    def check_simple(self):
        
        scores = PyMC2.geweke(sigma)
        print scores
        
class test_raftery_lewis(NumpyTestCase):
    def check_simple(self):
        print PyMC2.raftery_lewis(sigma, 0.5, .05)


if __name__ == "__main__":
    NumpyTest().run()
