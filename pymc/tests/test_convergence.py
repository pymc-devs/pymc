###
#
# Test of convergence diagnostics
#
###


from numpy.testing import assert_equal, assert_array_equal, TestCase
import unittest
import numpy as np
import pymc
import pymc.examples.weibull_fit as model

S = pymc.MCMC(model, 'ram')
S.sample(10000, 5000)
#a = S.a.trace()
#b = S.b.trace()

class test_geweke(TestCase):
    def test_simple(self):
        scores = pymc.geweke(S, intervals=20)
        a_scores = scores['a']
        assert_equal(len(a_scores), 20)
        
        # If the model has converged, 95% the scores should lie
        # within 2 standard deviations of zero, under standard normal model
        assert(sum(np.abs(np.array(a_scores)[:,1]) > 1.96) < 2)
        
        # Plot diagnostics (if plotting is available)
        try:
            from pymc.Matplot import geweke_plot as plot
            plot(scores)
        except ImportError:
            pass
        
class test_raftery_lewis(TestCase):
    def test_simple(self):

        nmin, kthin, nburn, nprec, kmind = pymc.raftery_lewis(S.a, 0.5, .05, verbose=0)
        
        # nmin should approximately be the same as nprec/kmind
        assert(0.8 < (float(nprec)/kmind) / nmin < 1.2)

if __name__ == "__main__":
    import unittest
    unittest.main()
