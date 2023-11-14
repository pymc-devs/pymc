#   Copyright 2023 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import collections

from typing import Callable, Dict

import numpy as np

__all__ = ["Callback", "CheckParametersConvergence", "ReduceLROnPlateau", "Tracker"]


class Callback:
    def __call__(self, approx, loss, i):
        raise NotImplementedError


def relative(current: np.ndarray, prev: np.ndarray, eps=1e-6) -> np.ndarray:
    diff = current - prev  # type: ignore
    return (np.abs(diff) + eps) / (np.abs(prev) + eps)


def absolute(current: np.ndarray, prev: np.ndarray) -> np.ndarray:
    diff = current - prev  # type: ignore
    return np.abs(diff)


_diff: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = dict(
    relative=relative, absolute=absolute
)


class CheckParametersConvergence(Callback):
    """Convergence stopping check

    Parameters
    ----------
    every: int
        check frequency
    tolerance: float
        if diff norm < tolerance: break
    diff: str
        difference type one of {'absolute', 'relative'}
    ord: {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
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

    def __init__(self, every=100, tolerance=1e-3, diff="relative", ord=np.inf):
        self._diff = _diff[diff]
        self.ord = ord
        self.every = every
        self.prev = None
        self.tolerance = tolerance

    def __call__(self, approx, _, i) -> None:
        if self.prev is None:
            self.prev = self.flatten_shared(approx.params)
            return
        if i % self.every or i < self.every:
            return
        current = self.flatten_shared(approx.params)
        prev = self.prev
        delta: np.ndarray = self._diff(current, prev)
        self.prev = current
        norm = np.linalg.norm(delta, self.ord)
        if norm < self.tolerance:
            raise StopIteration("Convergence achieved at %d" % i)

    @staticmethod
    def flatten_shared(shared_list):
        return np.concatenate([sh.get_value().flatten() for sh in shared_list])


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when the loss has stopped improving.

    This is inspired by Keras' homonymous callback:
    https://github.com/keras-team/keras/blob/v2.14.0/keras/callbacks.py

    Parameters
    ----------
    optimiser: callable
        PyMC optimiser
    factor: float
        factor by which the learning rate will be reduced: `new_lr = lr * factor`
    patience: int
        number of epochs with no improvement after which learning rate will be reduced
    min_lr: float
        lower bound on the learning rate
    cooldown: int
        number of iterations to wait before resuming normal operation after lr has been reduced
    verbose: bool
        false: quiet, true: update messages
    """

    def __init__(
        self,
        optimiser,
        factor=0.1,
        patience=10,
        min_lr=1e-6,
        cooldown=0,
    ):
        self.optimiser = optimiser
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.cooldown = cooldown

        self.cooldown_counter = 0
        self.wait = 0
        self.best = float("inf")
        self.old_lr = None

    def __call__(self, approx, loss_hist, i):
        current = loss_hist[-1]

        if np.isinf(current):
            return

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0
            return

        if current < self.best:
            self.best = current
            self.wait = 0
        elif not np.isinf(self.best):
            self.wait += 1
            if self.wait >= self.patience:
                self.reduce_lr()
                self.cooldown_counter = self.cooldown
                self.wait = 0

    def reduce_lr(self):
        old_lr = float(self.optimiser.keywords["learning_rate"])
        if old_lr > self.min_lr:
            new_lr = max(old_lr * self.factor, self.min_lr)
            self.optimiser.keywords["learning_rate"] = new_lr

    def in_cooldown(self):
        return self.cooldown_counter > 0


class Tracker(Callback):
    """
    Helper class to record arbitrary stats during VI

    It is possible to pass a function that takes no arguments
    If call fails then (approx, hist, i) are passed


    Parameters
    ----------
    kwargs: key word arguments
        keys mapping statname to callable that records the stat

    Examples
    --------
    Consider we want time on each iteration
    >>> import time
    >>> tracker = Tracker(time=time.time)
    >>> with model:
    ...     approx = pm.fit(callbacks=[tracker])

    Time can be accessed via :code:`tracker['time']` now
    For more complex summary one can use callable that takes
    (approx, hist, i) as arguments
    >>> with model:
    ...     my_callable = lambda ap, h, i: h[-1]
    ...     tracker = Tracker(some_stat=my_callable)
    ...     approx = pm.fit(callbacks=[tracker])

    Multiple stats are valid too
    >>> with model:
    ...     tracker = Tracker(some_stat=my_callable, time=time.time)
    ...     approx = pm.fit(callbacks=[tracker])
    """

    def __init__(self, **kwargs):
        self.whatchdict = kwargs
        self.hist = collections.defaultdict(list)

    def record(self, approx, hist, i):
        for key, fn in self.whatchdict.items():
            try:
                res = fn()
            # if `*t` argument is used
            # fail will be somehow detected.
            # We want both calls to be tried.
            # Upper one has more priority as
            # arbitrary functions can have some
            # defaults in positionals. Bad idea
            # to try fn(approx, hist, i) first
            except Exception:
                res = fn(approx, hist, i)
            self.hist[key].append(res)

    def clear(self):
        self.hist = collections.defaultdict(list)

    def __getitem__(self, item):
        return self.hist[item]

    __call__ = record
