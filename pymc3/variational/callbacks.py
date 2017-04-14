import numpy as np

__all__ = [
    'Callback',
    'CheckParametersConvergence'
]


class Callback(object):
    def __call__(self, approx, loss, i):
        raise NotImplementedError


class CheckParametersConvergence(Callback):
    def __init__(self, every=1000, tolerance=1e-2):
        self.every = every
        self.prev = None
        self.tolerance = tolerance

    def __call__(self, approx, _, i):
        if self.prev is None:
            self.prev = self.flatten_shared(approx.params)
        if i < self.every or i % self.every:
            return
        current = self.flatten_shared(approx.params)
        delta = (current - self.prev)/self.prev
        self.prev = current
        delta[np.isnan(delta)] = 0
        norm = delta.dot(delta)**.5
        if norm < self.tolerance:
            raise StopIteration

    @staticmethod
    def flatten_shared(shared_list):
        return np.concatenate([sh.get_value().flatten() for sh in shared_list])
