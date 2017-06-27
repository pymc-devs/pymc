import numpy as np
import theano.tensor as tt

from pymc3.distributions.distribution import (
    Distribution, Discrete, draw_values, generate_samples)
from pymc3.distributions import transforms
from pymc3.distributions.dist_math import bound

__all__ = ['Bound']


class _Bounded(Distribution):
    R"""
    An upper, lower or upper+lower bounded distribution

    Parameters
    ----------
    distribution : pymc3 distribution
        Distribution to be transformed into a bounded distribution
    lower : float (optional)
        Lower bound of the distribution, set to -inf to disable.
    upper : float (optional)
        Upper bound of the distribibution, set to inf to disable.
    tranform : 'infer' or object
        If 'infer', infers the right transform to apply from the supplied bounds.
        If transform object, has to supply .forward() and .backward() methods.
        See pymc3.distributions.transforms for more information.
    """

    def __init__(self, distribution, lower, upper,
                 transform='infer', *args, **kwargs):
        if lower == -np.inf:
            lower = None
        if upper == np.inf:
            upper = None

        if lower is not None:
            lower = tt.as_tensor_variable(lower)
        if upper is not None:
            upper = tt.as_tensor_variable(upper)

        self.lower = lower
        self.upper = upper

        if transform == 'infer':
            if lower is None and upper is None:
                transform = None
                default = None
            elif lower is not None and upper is not None:
                transform = transforms.interval(lower, upper)
                default = 0.5 * (lower + upper)
            elif upper is not None:
                transform = transforms.upperbound(upper)
                default = upper - 1
            else:
                transform = transforms.lowerbound(lower)
                default = lower + 1
        else:
            default = None

        # We don't use transformations for dicrete variables
        if issubclass(distribution, Discrete):
            transform = None

        kwargs['transform'] = transform
        self._wrapped = distribution.dist(*args, **kwargs)
        self._default = default

        if issubclass(distribution, Discrete) and default is not None:
            default = default.astype(str(self._wrapped.default().dtype))

        if default is None:
            defaults = self._wrapped.defaults
            for name in defaults:
                setattr(self, name, getattr(self._wrapped, name))
        else:
            defaults = ('_default',)

        super(_Bounded, self).__init__(
            shape=self._wrapped.shape,
            dtype=self._wrapped.dtype,
            testval=self._wrapped.testval,
            defaults=defaults,
            transform=self._wrapped.transform)

    def _random(self, lower, upper, point=None, size=None):
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        if lower.size > 1 or upper.size > 1:
            raise ValueError('Drawing samples from distributions with '
                             'array-valued bounds is not supported.')
        samples = np.zeros(size, dtype=self.dtype).flatten()
        i, n = 0, len(samples)
        while i < len(samples):
            sample = self._wrapped.random(point=point, size=n)
            select = sample[np.logical_and(sample >= lower, sample <= upper)]
            samples[i:(i + len(select))] = select[:]
            i += len(select)
            n -= len(select)
        if size is not None:
            return np.reshape(samples, size)
        else:
            return samples

    def random(self, point=None, size=None, repeat=None):
        if self.lower is None and self.upper is None:
            return self._wrapped.random(point=point, size=size)
        elif self.lower is not None and self.upper is not None:
            lower, upper = draw_values([self.lower, self.upper], point=point)
            return generate_samples(self._random, lower, upper, point,
                                    dist_shape=self.shape,
                                    size=size)
        elif self.lower is not None:
            lower = draw_values([self.lower], point=point)
            return generate_samples(self._random, lower, np.inf, point,
                                    dist_shape=self.shape,
                                    size=size)
        else:
            upper = draw_values([self.upper], point=point)
            return generate_samples(self._random, -np.inf, upper, point,
                                    dist_shape=self.shape,
                                    size=size)

    def logp(self, value):
        logp = self._wrapped.logp(value)
        bounds = []
        if self.lower is not None:
            bounds.append(value >= self.lower)
        if self.upper is not None:
            bounds.append(value <= self.upper)
        if len(bounds) > 0:
            return bound(logp, *bounds)
        else:
            return logp


class Bound(object):
    R"""
    Create a new upper, lower or upper+lower bounded distribution.

    The resulting distribution is not normalized anymore. This
    is usually fine if the bounds are constants. If you need
    truncated distributions, use `Bound` in combination with
    a `pm.Potential` with the cumulative probability function.

    The bounds are inclusive for discrete distributions.

    Parameters
    ----------
    distribution : pymc3 distribution
        Distribution to be transformed into a bounded distribution.
    lower : float or array like, optional
        Lower bound of the distribution.
    upper : float or array like, optional
        Upper bound of the distribution.

    Example
    -------
    # Bounded distribution can be defined before the model context
    PositiveNormal = pm.Bound(pm.Normal, lower=0.0)
    with pm.Model():
        par1 = PositiveNormal('par1', mu=0.0, sd=1.0, testval=1.0)
        # or within the model context
        NegativeNormal = pm.Bound(pm.Normal, upper=0.0)
        par2 = NegativeNormal('par2', mu=0.0, sd=1.0, testval=1.0)

        # or you can define it implicitly within the model context
        par3 = pm.Bound(pm.Normal, lower=-1.0, upper=1.0)(
                'par3', mu=0.0, sd=1.0, testval=1.0)
    """

    def __init__(self, distribution, lower=None, upper=None):
        self.distribution = distribution
        self.lower = lower
        self.upper = upper

    def __call__(self, *args, **kwargs):
        if 'observed' in kwargs:
            raise ValueError('Observed Bound distributions are not allowed. '
                             'If you want to model truncated data '
                             'you can use a pm.Potential in combination '
                             'with the cumulative probability function. See '
                             'pymc3/examples/censored_data.py for an example.')
        first, args = args[0], args[1:]

        return _Bounded(first, self.distribution, self.lower, self.upper,
                        *args, **kwargs)

    def dist(self, *args, **kwargs):
        return _Bounded.dist(self.distribution, self.lower, self.upper,
                             *args, **kwargs)
