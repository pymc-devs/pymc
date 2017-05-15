import numpy as np

__all__ = [
    'Callback',
    'CheckParametersConvergence'
]


class Callback(object):
    def __call__(self, approx, loss, i):
        raise NotImplementedError


def relative(current, prev, eps=1e-6):
    return (np.abs(current - prev) + eps) / (np.abs(prev) + eps)


def absolute(current, prev):
    return np.abs(current - prev)


_diff = dict(
    relative=relative,
    absolute=absolute
)


class CheckParametersConvergence(Callback):
    """Convergence stopping check

    Parameters
    ----------
    every : int
        check frequency
    tolerance : float
        if diff norm < tolerance : break
    diff : str
        difference type one of {'absolute', 'relative'}
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        see more info in :func:`numpy.linalg.norm`

    Examples
    --------
    >>> with model:
    ...     approx = pm.fit(
    ...         n=10000, callbacks=[
    ...             CheckParametersConvergence(
    ...                 every=50, diff='absolute',
    ...                 tolerance=1e-4)
    ...         ]
    ...     )
    """

    def __init__(self, every=100, tolerance=1e-3,
                 diff='relative', ord=np.inf):
        self._diff = _diff[diff]
        self.ord = ord
        self.every = every
        self.prev = None
        self.tolerance = tolerance

    def __call__(self, approx, _, i):
        if self.prev is None:
            self.prev = self.flatten_shared(approx.params)
            return
        if i % self.every or i < self.every:
            return
        current = self.flatten_shared(approx.params)
        prev = self.prev
        delta = self._diff(current, prev)  # type: np.ndarray
        self.prev = current
        norm = np.linalg.norm(delta, self.ord)
        if norm < self.tolerance:
            raise StopIteration('Convergence archived at %d' % i)

    @staticmethod
    def flatten_shared(shared_list):
        return np.concatenate([sh.get_value().flatten() for sh in shared_list])
