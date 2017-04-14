import scipy.stats as stats
import numpy as np

__all__ = [
    'Callback',
    'CheckLossConvergence1',
    'CheckLossConvergence2'
]


class Callback(object):
    def __call__(self, approx, loss, i):
        raise NotImplementedError


class CheckLossConvergence1(Callback):
    def __init__(self, every=100, window_size=2000, tolerance=1e-3):
        """

        Parameters
        ----------
        every : int
            how often check convergence
        window_size :
            last elbos to take
        tolerance : float
            Error rate under null hypothesis, consider taking small values
        """
        self.every = every
        self.window_size = window_size
        self.critical = tolerance / 2.

    def __call__(self, approx, hist, i):
        if hist is None or i < self.window_size or i % self.every:
            return
        diff = ((hist[-self.window_size:] - hist[-self.window_size-1:-1])
                / hist[-self.window_size-1:-1])
        mean = diff.mean()
        # unbiased std of mean
        std = diff.std() / (self.window_size - 1)**.5
        t = abs(mean / std)
        p = stats.t.cdf(t, df=self.window_size) - .5
        # 1 - confidence is lower allowed p
        if p < self.critical:
            raise StopIteration


class CheckLossConvergence2(Callback):
    def __init__(self, every=100, tolerance=1e-2, steps=None):
        self.steps = steps
        self.every = every
        self.tolerance = tolerance

    def __call__(self, approx, hist, i):
        if hist is None or i < self.every or i % self.every:
            return
        if self.steps is None:
            window = int(max(0.1 * hist.size // self.every, 2.0))
        else:
            window = int(max(0.1 * self.steps // self.every, 2.0))
        losses = hist[::self.every][-window:]
        diff = np.abs((losses[1:]-losses[:-1])/losses[:-1])
        mean = np.mean(diff)
        med = np.median(diff)
        if mean < self.tolerance or med < self.tolerance:
            raise StopIteration
