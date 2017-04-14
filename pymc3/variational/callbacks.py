import scipy.stats as stats


class Callback(object):
    def __call__(self, approx, loss, i):
        raise NotImplementedError


class CheckLossConvergence(Callback):
    def __init__(self, every=100, window_size=1000, tolerance=1e-3):
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
