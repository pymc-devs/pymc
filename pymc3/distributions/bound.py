from numbers import Real

import numpy as np
import theano.tensor as tt
import theano

from pymc3.distributions.distribution import (
    Distribution, Discrete, Continuous, draw_values, generate_samples)
from pymc3.distributions import transforms
from pymc3.distributions.dist_math import bound

__all__ = ['Bound']


class _Bounded(Distribution):
    def __init__(self, distribution, lower, upper, default, *args, **kwargs):
        self.lower = lower
        self.upper = upper
        self._wrapped = distribution.dist(*args, **kwargs)

        if default is None:
            defaults = self._wrapped.defaults
            for name in defaults:
                setattr(self, name, getattr(self._wrapped, name))
        else:
            defaults = ('_default',)
            self._default = default

        super().__init__(
            shape=self._wrapped.shape,
            dtype=self._wrapped.dtype,
            testval=self._wrapped.testval,
            defaults=defaults,
            transform=self._wrapped.transform)

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

    def _random(self, lower, upper, point=None, size=None):
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        if lower.size > 1 or upper.size > 1:
            raise ValueError('Drawing samples from distributions with '
                             'array-valued bounds is not supported.')
        samples = np.zeros(size, dtype=self.dtype).flatten()
        i, n = 0, len(samples)
        while i < len(samples):
            sample = np.atleast_1d(self._wrapped.random(point=point, size=n))

            select = sample[np.logical_and(sample >= lower, sample <= upper)]
            samples[i:(i + len(select))] = select[:]
            i += len(select)
            n -= len(select)
        if size is not None:
            return np.reshape(samples, size)
        else:
            return samples

    def random(self, point=None, size=None):
        if self.lower is None and self.upper is None:
            return self._wrapped.random(point=point, size=size)
        elif self.lower is not None and self.upper is not None:
            lower, upper = draw_values([self.lower, self.upper], point=point, size=size)
            return generate_samples(self._random, lower, upper, point,
                                    dist_shape=self.shape,
                                    size=size)
        elif self.lower is not None:
            lower = draw_values([self.lower], point=point, size=size)
            return generate_samples(self._random, lower, np.inf, point,
                                    dist_shape=self.shape,
                                    size=size)
        else:
            upper = draw_values([self.upper], point=point, size=size)
            return generate_samples(self._random, -np.inf, upper, point,
                                    dist_shape=self.shape,
                                    size=size)


class _DiscreteBounded(_Bounded, Discrete):
    def __init__(self, distribution, lower, upper,
                 transform='infer', *args, **kwargs):
        if transform == 'infer':
            transform = None
        if transform is not None:
            raise ValueError('Can not transform discrete variable.')

        if lower is None and upper is None:
            default = None
        elif lower is not None and upper is not None:
            default = (lower + upper) // 2
        if upper is not None:
            default = upper - 1
        if lower is not None:
            default = lower + 1

        super().__init__(
            distribution=distribution, lower=lower, upper=upper,
            default=default, *args, **kwargs)


class _ContinuousBounded(_Bounded, Continuous):
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
        dtype = kwargs.get('dtype', theano.config.floatX)

        if lower is not None:
            lower = tt.as_tensor_variable(lower).astype(dtype)
        if upper is not None:
            upper = tt.as_tensor_variable(upper).astype(dtype)

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

        super().__init__(
            distribution=distribution, lower=lower, upper=upper,
            transform=transform, default=default, *args, **kwargs)


class Bound:
    R"""
    Create a Bound variable object that can be applied to create
    a new upper, lower, or upper and lower bounded distribution.

    The resulting distribution is not normalized anymore. This
    is usually fine if the bounds are constants. If you need
    truncated distributions, use `Bound` in combination with
    a :class:`~pymc3.model.Potential` with the cumulative probability function.

    The bounds are inclusive for discrete distributions.

    Parameters
    ----------
    distribution : pymc3 distribution
        Distribution to be transformed into a bounded distribution.
    lower : float or array like, optional
        Lower bound of the distribution.
    upper : float or array like, optional
        Upper bound of the distribution.

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            NegativeNormal = pm.Bound(pm.Normal, upper=0.0)
            par1 = NegativeNormal('par`', mu=0.0, sigma=1.0, testval=-0.5)
            # you can use the Bound object multiple times to
            # create multiple bounded random variables
            par1_1 = NegativeNormal('par1_1', mu=-1.0, sigma=1.0, testval=-1.5)

            # you can also define a Bound implicitly, while applying
            # it to a random variable
            par2 = pm.Bound(pm.Normal, lower=-1.0, upper=1.0)(
                    'par2', mu=0.0, sigma=1.0, testval=1.0)
    """

    def __init__(self, distribution, lower=None, upper=None):
        if isinstance(lower, Real) and lower == -np.inf:
            lower = None
        if isinstance(upper, Real) and upper == np.inf:
            upper = None

        if not issubclass(distribution, Distribution):
            raise ValueError('"distribution" must be a Distribution subclass.')

        self.distribution = distribution
        self.lower = lower
        self.upper = upper

    def __call__(self, name, *args, **kwargs):
        if 'observed' in kwargs:
            raise ValueError('Observed Bound distributions are not supported. '
                             'If you want to model truncated data '
                             'you can use a pm.Potential in combination '
                             'with the cumulative probability function. See '
                             'pymc3/examples/censored_data.py for an example.')

        if issubclass(self.distribution, Continuous):
            return _ContinuousBounded(name, self.distribution,
                                      self.lower, self.upper, *args, **kwargs)
        elif issubclass(self.distribution, Discrete):
            return _DiscreteBounded(name, self.distribution,
                                    self.lower, self.upper, *args, **kwargs)
        else:
            raise ValueError(
                'Distribution is neither continuous nor discrete.')

    def dist(self, *args, **kwargs):
        if issubclass(self.distribution, Continuous):
            return _ContinuousBounded.dist(
                self.distribution, self.lower, self.upper, *args, **kwargs)

        elif issubclass(self.distribution, Discrete):
            return _DiscreteBounded.dist(
                self.distribution, self.lower, self.upper, *args, **kwargs)
        else:
            raise ValueError('Distribution is neither continuous nor discrete.')
