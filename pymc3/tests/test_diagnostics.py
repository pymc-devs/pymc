import unittest

from numpy.testing import assert_allclose, assert_array_less

from ..model import Model
from ..step_methods import Slice, Metropolis, NUTS
from ..distributions import Normal
from ..tuning import find_MAP
from ..sampling import sample
from ..diagnostics import effective_n, geweke, gelman_rubin
from .test_examples import build_disaster_model


class TestGelmanRubin(unittest.TestCase):
    good_ratio = 1.1

    def get_ptrace(self, n_samples):
        model = build_disaster_model()
        with model:
            # Run sampler
            step1 = Slice([model.early_mean_log_, model.late_mean_log_])
            step2 = Metropolis([model.switchpoint])
            start = {'early_mean': 2., 'late_mean': 3., 'switchpoint': 50}
            ptrace = sample(n_samples, [step1, step2], start, njobs=2, progressbar=False,
                            random_seed=[1, 3])
        return ptrace

    def test_good(self):
        """Confirm Gelman-Rubin statistic is close to 1 for a reasonable number of samples."""
        n_samples = 1000
        rhat = gelman_rubin(self.get_ptrace(n_samples))
        self.assertTrue(all(1 / self.good_ratio < r <
                            self.good_ratio for r in rhat.values()))

    def test_bad(self):
        """Confirm Gelman-Rubin statistic is far from 1 for a small number of samples."""
        n_samples = 10
        rhat = gelman_rubin(self.get_ptrace(n_samples))
        self.assertFalse(all(1 / self.good_ratio < r <
                             self.good_ratio for r in rhat.values()))


class TestDiagnostics(unittest.TestCase):

    def get_switchpoint(self, n_samples):
        model = build_disaster_model()
        with model:
            # Run sampler
            step1 = Slice([model.early_mean_log_, model.late_mean_log_])
            step2 = Metropolis([model.switchpoint])
            trace = sample(n_samples, [step1, step2], progressbar=False, random_seed=1)
        return trace['switchpoint']

    def test_geweke_negative(self):
        """Confirm Geweke diagnostic is larger than 1 for a small number of samples."""
        n_samples = 200
        n_intervals = 20
        switchpoint = self.get_switchpoint(n_samples)
        first = 0.1
        last = 0.7
        # returns (intervalsx2) matrix, with first row start indexes, second
        # z-scores
        z_switch = geweke(switchpoint, first=first,
                          last=last, intervals=n_intervals)

        # These z-scores should be larger, since there are not many samples.
        self.assertGreater(max(abs(z_switch[:, 1])), 1)

    def test_geweke_positive(self):
        """Confirm Geweke diagnostic is smaller than 1 for a reasonable number of samples."""
        n_samples = 2000
        n_intervals = 20
        switchpoint = self.get_switchpoint(n_samples)

        with self.assertRaises(ValueError):
            # first and last must be between 0 and 1
            geweke(switchpoint, first=-0.3, last=1.1, intervals=n_intervals)

        with self.assertRaises(ValueError):
            # first and last must add to < 1
            geweke(switchpoint, first=0.3, last=0.7, intervals=n_intervals)

        first = 0.1
        last = 0.7
        # returns (intervalsx2) matrix, with first row start indexes, second
        # z-scores
        z_switch = geweke(switchpoint, first=first,
                          last=last, intervals=n_intervals)
        start = z_switch[:, 0]
        z_scores = z_switch[:, 1]

        # Ensure `intervals` argument is honored
        self.assertEqual(z_switch.shape[0], n_intervals)

        # Start index should not be in the last <last>% of samples
        assert_array_less(start, (1 - last) * n_samples)

        # These z-scores should be small, since there are more samples.
        self.assertLess(max(abs(z_scores)), 1)

    def test_effective_n(self):
        """Check effective sample size is equal to number of samples when initializing with MAP"""
        n_jobs = 3
        n_samples = 100

        with Model():
            Normal('x', 0, 1., shape=5)

            # start sampling at the MAP
            start = find_MAP()
            step = NUTS(scaling=start)
            ptrace = sample(n_samples, step, start,
                            njobs=n_jobs, random_seed=42)

        n_effective = effective_n(ptrace)['x']
        assert_allclose(n_effective, n_jobs * n_samples, 2)
