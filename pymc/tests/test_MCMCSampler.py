"""
The DisasterMCMC example.

"""
from numpy.testing import *
from pymc import MCMC
from pymc.examples import DisasterModel
import nose,  warnings
warnings.simplefilter('ignore', FutureWarning)
PLOT=True
try:
    from pymc.Matplot import plot, autocorrelation
except:
    PLOT=False
    pass


DIR = 'testresults/'


class test_MCMC(TestCase):

    # Instantiate samplers
    M = MCMC(DisasterModel)

    # Sample
    M.sample(4000,2000,verbose=0)

    def test_instantiation(self):

        # Check stochastic arrays
        assert_equal(len(self.M.stochastics), 3)
        assert_equal(len(self.M.observed_stochastics),1)
        assert_array_equal(self.M.D.value, DisasterModel.disasters_array)

    def test_plot(self):
        if not PLOT:
            raise nose.SkipTest

        # Plot samples
        plot(self.M.e, path=DIR, verbose=0)

    def test_autocorrelation(self):
        if not PLOT:
            raise nose.SkipTest

        # Plot samples
        autocorrelation(self.M.e, path=DIR,  verbose=0)

    def test_stats(self):
        S = self.M.e.stats()


if __name__ == '__main__':
    warnings.simplefilter('ignore',  FutureWarning)
    nose.runmodule()

