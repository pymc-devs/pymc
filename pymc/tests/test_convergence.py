#
#
# Test of convergence diagnostics
#
#

from __future__ import with_statement
from numpy.testing import assert_equal, assert_array_equal
from numpy.testing import assert_approx_equal, TestCase
import unittest
import numpy as np
import pymc
import pymc.examples.weibull_fit as model
import os
import warnings
import copy

np.random.seed(467)

S = pymc.MCMC(model, 'ram')
S.sample(10000, 2000, progress_bar=0)
# a = S.a.trace()
# b = S.b.trace()

# Known data for testing integrated autocorrelation time = 2.28
x = np.array([0.98073604, 0.98073604, 0.98073604, 0.98073604, 0.98073604,
              0.41424798, 0.58398493, 0.27391045, 0.27391045, 0.27391045,
              0.27391045, 0.27391045, 0.72886149, 0.72886149, 0.72886149,
              0.67478139, 0.67478139, 0.67478139, 0.67478139, 0.67478139,
              0.67478139, 0.27720909, 0.6026456, 0.6026456, 0.47108579,
              0.47108579, 0.47108579, 0.47108579, 0.47108579, 0.47108579,
              0.47108579, 0.47108579, 0.47108579, 0.47108579, 0.47108579,
              0.47108579, 0.47108579, 0.47108579, 0.47108579, 0.47108579,
              0.47108579, 0.47108579, 0.47108579, 0.47108579, 0.47108579,
              0.34546653, 0.34546653, 0.5441314, 0.5441314, 0.5441314,
              0.5441314, 0.5441314, 0.5441314, 0.5441314, 0.5441314,
              0.37344506, 0.37344506, 0.83126209, 0.83126209, 0.3439339,
              0.3439339, 0.3439339, 0.34551721, 0.34551721, 0.34551721,
              0.44112754, 0.44112754, 0.44112754, 0.55397635, 0.55397635,
              0.55397635, 0.55397635, 0.55397635, 0.55397635, 0.55397635,
              0.55397635, 0.55397635, 0.55397635, 0.55397635, 0.55397635,
              0.55397635, 0.55397635, 0.55397635, 0.55397635, 0.55397635,
              0.55397635, 0.36521137, 0.36521137, 0.36521137, 0.36521137,
              0.36521137, 0.36521137, 0.36521137, 0.36521137, 0.36521137,
              0.36521137, 0.36521137, 0.36521137, 0.36521137, 0.32755692])


DIR = 'testresults'


class test_geweke(TestCase):

    def test_independent(self):
        # Use IID data

        x = [pymc.geweke(np.random.normal(size=1000), intervals=5)[0][1] for _ in range(10000)]

        assert_approx_equal(np.var(x), 1, 1)

        # If the model has converged, 95% the scores should lie
        # within 2 standard deviations of zero, under standard normal model
        intervals = 40
        x = np.transpose(pymc.geweke(np.random.normal(size=10000), intervals=intervals))[1]
        assert(sum(np.abs(x) < 2) >= int(0.9 * intervals))

    def test_simple(self):

        intervals = 20
        scores = pymc.geweke(S, intervals=intervals)
        a_scores = scores['a']
        assert_equal(len(a_scores), intervals)

        # Plot diagnostics (if plotting is available)
        try:
            from pymc.Matplot import geweke_plot as plot
            plot(scores, path=DIR, verbose=0)
        except ImportError:
            pass


class test_gelman_rubin(TestCase):

    """Unit test for Gelman-Rubin diagnostic"""

    def test_fail(self):

        pass

    def test_simple(self):

        S2 = copy.copy(S)
        S2.sample(10000, 2000, progress_bar=0)

        gr = pymc.gelman_rubin(S2)

        for i in gr:
            assert_approx_equal(gr[i], 1., 2)


class test_iat(TestCase):

    def test_simple(self):

        iat = pymc.iat(x)

        # IAT should be approximately 2.28
        assert_approx_equal(iat, 2.28, 2)

if __name__ == "__main__":

    original_filters = warnings.filters[:]
    warnings.simplefilter("ignore")
    try:
        import nose
        C = nose.config.Config(verbosity=1)
        nose.runmodule(config=C)
    finally:
        warnings.filters = original_filters

    # TODO: Restore in 2.2
    # with warnings.catch_warnings():
    #         warnings.simplefilter('ignore', FutureWarning)
    #         import nose
    #         C =nose.config.Config(verbosity=1)
    #         nose.runmodule(config=C)
