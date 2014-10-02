"""
The DisasterMCMC example.

"""
from __future__ import with_statement
from numpy.testing import *
from pymc import MCMC, database
from pymc.examples import disaster_model
import nose
import warnings
import os

PLOT = True
try:
    from pymc.Matplot import plot, autocorrelation
except:
    PLOT = False
    pass


DIR = 'testresults/'


class test_tiny_MCMC(TestCase):

    # Instantiate samplers
    M = MCMC(disaster_model)

    # Sample
    M.sample(10, progress_bar=False)

    def test_plot(self):

        if not PLOT:
            raise nose.SkipTest

        # Plot samples
        plot(self.M, path=DIR, verbose=0)


class test_MCMC(TestCase):

    # Instantiate samplers
    M = MCMC(disaster_model, db='pickle')

    # Sample
    M.sample(2000, 100, thin=15, verbose=0, progress_bar=False)
    M.db.close()

    def test_instantiation(self):

        # Check stochastic arrays
        assert_equal(len(self.M.stochastics), 3)
        assert_equal(len(self.M.observed_stochastics), 1)
        assert_array_equal(
            self.M.disasters.value,
            disaster_model.disasters_array)

    def test_plot(self):
        if not PLOT:
            raise nose.SkipTest

        # Plot samples
        plot(self.M.early_mean, path=DIR, verbose=0)

    def test_autocorrelation(self):
        if not PLOT:
            raise nose.SkipTest

        # Plot samples
        autocorrelation(self.M.early_mean, path=DIR, verbose=0)

    def test_stats(self):
        S = self.M.early_mean.stats()
        self.M.stats()
        
    def test_summary(self):
        self.M.rate.summary()

    def test_stats_after_reload(self):
        db = database.pickle.load('MCMC.pickle')
        M2 = MCMC(disaster_model, db=db)
        M2.stats()
        db.close()
        os.remove('MCMC.pickle')


if __name__ == '__main__':

    original_filters = warnings.filters[:]
    warnings.simplefilter("ignore")
    try:
        import nose
        nose.runmodule()
    finally:
        warnings.filters = original_filters

    # TODO: Restore in 2.2
    # with warnings.catch_warnings():
    #         warnings.simplefilter('ignore',  FutureWarning)
    #         nose.runmodule()
