import numpy as np

__all__ = [
    'Callback',
    'CheckParametersConvergence'
]


class Callback(object):
    def __call__(self, approx, loss, i):
        raise NotImplementedError


class CheckParametersConvergence(Callback):
    def __init__(self, every=1000, tolerance=1e-3, eps=1e-10):
        self.every = every
        self.prev = None
        self.tolerance = tolerance
        self.eps = np.float32(eps)

    def __call__(self, approx, _, i):
        if self.prev is None:
            self.prev = self.flatten_shared(approx.params)
        if i < self.every or i % self.every:
            return
        current = self.flatten_shared(approx.params)
        prev = self.prev
        eps = self.eps
        delta = (np.abs(current - prev)+eps)/(np.abs(prev)+eps)
        self.prev = current
        norm = delta.max()
        if norm < self.tolerance:
            raise StopIteration('Convergence archived')

    @staticmethod
    def flatten_shared(shared_list):
        return np.concatenate([sh.get_value().flatten() for sh in shared_list])
