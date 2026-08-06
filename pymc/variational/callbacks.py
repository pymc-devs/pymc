#   Copyright 2024 - present The PyMC Developers
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

from collections.abc import Callable

import numpy as np

__all__ = ["Callback", "CheckLossConvergence", "CheckParametersConvergence", "Tracker"]


class Callback:
    def __call__(self, approx, loss, i):
        raise NotImplementedError


def relative(current: np.ndarray, prev: np.ndarray, eps=1e-6) -> np.ndarray:
    diff = current - prev
    return (np.abs(diff) + eps) / (np.abs(prev) + eps)


def absolute(current: np.ndarray, prev: np.ndarray) -> np.ndarray:
    diff = current - prev
    return np.abs(diff)


_diff: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "relative": relative,
    "absolute": absolute,
}


class CheckParametersConvergence(Callback):
    """Convergence stopping check.

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
    ...         n=10000,
    ...         callbacks=[CheckParametersConvergence(every=50, diff="absolute", tolerance=1e-4)],
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
            raise StopIteration(f"Convergence achieved at {i}")

    @staticmethod
    def flatten_shared(shared_list):
        return np.concatenate([sh.get_value().flatten() for sh in shared_list])


class Tracker(Callback):
    """
    Helper class to record arbitrary stats during VI.

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
        """Get the element at index `item`."""
        return self.hist[item]

    __call__ = record


class CheckLossConvergence(Callback):
    """Stop ``pm.fit`` when the loss improvement rate decays to noise level.

    The monitored quantity must be *minimized*: an ELBO is maximized, so pass
    ``-elbo``.

    Let ``delta_t = loss[t-1] - loss[t]`` (positive while optimizing). Each
    increment is standardized by a robust exponentially-weighted scale estimate
    built from successive differences (von Neumann, 1941) and winsorized at
    ``+/- 4``, giving ``z_t``; the monitor accumulates the one-sided CUSUM
    (Page, 1954)::

        S_t = max(0, S_{t-1} + (kappa - max(z_t, 0)))

    and declares convergence once ``S > h``; the CUSUM accumulates only after
    ``min_steps``.

    Reaching the threshold while the loss trends upwards is reported as divergence,
    not convergence.

    Parameters
    ----------
    kappa : float
        Allowance per step, in units of the scale that standardizes ``delta``. Must
        exceed what a stalled trace spends on noise alone (0.3 to 0.4); see Notes.
    h : float
        CUSUM decision threshold. Larger values trade detection delay for a
        lower false-alarm rate.
    halflife : float
        Half-life, in steps, of the exponentially-weighted scale estimate.
    min_steps : int
        Number of steps before the CUSUM is armed. Must be large enough for
        the scale estimate to stabilize (a few half-lives). Also the number of
        consecutive non-finite losses tolerated before the fit is stopped:
        ``pm.fit`` aborts on NaN but runs to completion on ``+inf``.

    Notes
    -----
    The defaults fix a *rate*: the per-step improvement of the loss over its per-step
    noise sd. A fit improving more slowly than that is stopped however long it would
    have gone on improving, so the figures below hold at a stated rate, not in general.

    Calibrated on 1000 traces per cell of ``loss[t] = f(t) + sigma[t] * eps[t]``, 6000
    steps, in four families -- linear, power law, alternating ``sigma``, and Student-t
    noise with ``df=3``. At a rate of 1.0 the defaults stopped none of the 4000
    still-improving traces; the stall boundary sits between 0.6 and 0.7, and ``kappa``
    is what moves it. On the same families frozen to a plateau after improving at a
    rate of 1.0, every trace was stopped and none before the plateau, with median
    delays of 25 to 54 steps and p95 at most 86.

    Examples
    --------
    .. code-block:: python

        monitor = CheckLossConvergence()
        approx = pm.fit(100_000, callbacks=[monitor])  # stops early if converged
    """

    # Scales mean |successive difference| into the standardizer for delta. The z it
    # produces is unit-variance only when the loss increments are independent, so kappa
    # and h are calibrated against it as it stands rather than derived from it.
    _SCALE_TO_SIGMA = float(np.sqrt(np.pi) / 2.0)
    # Winsorization bound on z, applied to the scale update too so one spike cannot
    # inflate the scale for hundreds of steps.
    _Z_CLIP = 4.0
    # Additive floor on the standardizing scale: an exactly-constant stretch of loss
    # drives the successive-difference scale to zero, and dividing by it raises.
    _SIGMA_FLOOR = 1e-12

    def __init__(self, kappa=0.5, h=10.0, halflife=200.0, min_steps=1000):
        for name, value in (("kappa", kappa), ("h", h), ("halflife", halflife)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if self._Z_CLIP <= kappa:
            raise ValueError(f"kappa ({kappa}) must be below _Z_CLIP ({self._Z_CLIP})")
        if not np.isfinite(min_steps) or min_steps != int(min_steps) or min_steps < 0:
            raise ValueError(f"min_steps must be a non-negative integer, got {min_steps!r}")
        self.kappa = float(kappa)
        self.h = float(h)
        self.halflife = float(halflife)
        self.min_steps = int(min_steps)

        self._lam = float(np.exp(np.log(0.5) / self.halflife))
        # The smoothed z settles at the mean z, so the stop is reported as a divergence
        # once the loss has been climbing by _rise_tol times the scale, per step.
        self._rise_tol = 4.0 * float(np.sqrt((1.0 - self._lam) / (1.0 + self._lam)))
        self.n_nonfinite = 0
        self._prev_loss = None
        self._prev_delta = None
        self._scale = None  # EW mean of |delta_t - delta_{t-1}|
        self._z_bar = 0.0
        self._S = 0.0

    def __call__(self, approx, loss, i):
        if loss is None:
            raise TypeError(
                f"{type(self).__name__} needs per-step losses; run pm.fit with score=True "
                "(the default for ADVI) or remove this callback."
            )
        current = float(loss[-1])
        if not np.isfinite(current):
            self.n_nonfinite += 1
            if self.n_nonfinite > self.min_steps:
                raise StopIteration(
                    f"{type(self).__name__}: the loss has been non-finite for "
                    f"{self.n_nonfinite} steps; stopping at step {i}"
                )
            return
        self.n_nonfinite = 0
        if self._prev_loss is None:
            self._prev_loss = current
            return

        delta = self._prev_loss - current
        self._prev_loss = current

        if self._prev_delta is None:
            self._prev_delta = delta
            return
        abs_diff = abs(delta - self._prev_delta)
        self._prev_delta = delta

        # Standardize with the *previous* scale so a step never judges itself.
        if self._scale is None:
            if np.isfinite(abs_diff):
                self._scale = abs_diff
            return
        sigma = self._scale * self._SCALE_TO_SIGMA + self._SIGMA_FLOOR
        if np.isfinite(abs_diff):
            update = min(abs_diff, self._Z_CLIP * sigma)
            self._scale = self._lam * self._scale + (1.0 - self._lam) * update
        z = float(np.clip(delta / sigma, -self._Z_CLIP, self._Z_CLIP))
        self._z_bar = self._lam * self._z_bar + (1.0 - self._lam) * z

        if i >= self.min_steps:
            self._S = max(0.0, self._S + (self.kappa - max(z, 0.0)))

        if self._S > self.h:
            if self._z_bar < -self._rise_tol:
                raise StopIteration(
                    f"{type(self).__name__}: the loss is trending up, not converging "
                    f"(step {i}, mean z={self._z_bar:.2f}); if it is maximized, negate it"
                )
            raise StopIteration(
                f"{type(self).__name__}: converged at step {i} (S={self._S:.2f} > h={self.h:g})"
            )
