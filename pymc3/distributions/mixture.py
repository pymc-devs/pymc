import numpy as np
import theano.tensor as tt

from pymc3.util import get_variable_name
from ..math import logsumexp
from .dist_math import bound, random_choice
from .distribution import (Discrete, Distribution, draw_values,
                           generate_samples, _DrawValuesContext,
                           _DrawValuesContextBlocker, to_tuple)
from .continuous import get_tau_sigma, Normal


def all_discrete(comp_dists):
    """
    Determine if all distributions in comp_dists are discrete
    """
    if isinstance(comp_dists, Distribution):
        return isinstance(comp_dists, Discrete)
    else:
        return all(isinstance(comp_dist, Discrete) for comp_dist in comp_dists)


class Mixture(Distribution):
    R"""
    Mixture log-likelihood

    Often used to model subpopulation heterogeneity

    .. math:: f(x \mid w, \theta) = \sum_{i = 1}^n w_i f_i(x \mid \theta_i)

    ========  ============================================
    Support   :math:`\cap_{i = 1}^n \textrm{support}(f_i)`
    Mean      :math:`\sum_{i = 1}^n w_i \mu_i`
    ========  ============================================

    Parameters
    ----------
    w : array of floats
        w >= 0 and w <= 1
        the mixture weights
    comp_dists : multidimensional PyMC3 distribution (e.g. `pm.Poisson.dist(...)`)
        or iterable of one-dimensional PyMC3 distributions the
        component distributions :math:`f_1, \ldots, f_n`

    Example
    -------
    .. code-block:: python

        # 2-Mixture Poisson distribution
        with pm.Model() as model:
            lam = pm.Exponential('lam', lam=1, shape=(2,))  # `shape=(2,)` indicates two mixtures.

            # As we just need the logp, rather than add a RV to the model, we need to call .dist()
            components = pm.Poisson.dist(mu=lam, shape=(2,))

            w = pm.Dirichlet('w', a=np.array([1, 1]))  # two mixture component weights.

            like = pm.Mixture('like', w=w, comp_dists=components, observed=data)

        # 2-Mixture Poisson using iterable of distributions.
        with pm.Model() as model:
            lam1 = pm.Exponential('lam1', lam=1)
            lam2 = pm.Exponential('lam2', lam=1)

            pois1 = pm.Poisson.dist(mu=lam1)
            pois2 = pm.Poisson.dist(mu=lam2)

            w = pm.Dirichlet('w', a=np.array([1, 1]))

            like = pm.Mixture('like', w=w, comp_dists = [pois1, pois2], observed=data)
    """

    def __init__(self, w, comp_dists, *args, **kwargs):
        shape = kwargs.pop('shape', ())

        self.w = w = tt.as_tensor_variable(w)
        self.comp_dists = comp_dists

        defaults = kwargs.pop('defaults', [])

        if all_discrete(comp_dists):
            dtype = kwargs.pop('dtype', 'int64')
        else:
            dtype = kwargs.pop('dtype', 'float64')

            try:
                self.mean = (w * self._comp_means()).sum(axis=-1)

                if 'mean' not in defaults:
                    defaults.append('mean')
            except AttributeError:
                pass

        try:
            comp_modes = self._comp_modes()
            comp_mode_logps = self.logp(comp_modes)
            self.mode = comp_modes[tt.argmax(w * comp_mode_logps, axis=-1)]

            if 'mode' not in defaults:
                defaults.append('mode')
        except (AttributeError, ValueError, IndexError):
            pass

        super().__init__(shape, dtype, defaults=defaults, *args, **kwargs)

    @property
    def comp_dists(self):
        return self._comp_dists

    @comp_dists.setter
    def comp_dists(self, _comp_dists):
        self._comp_dists = _comp_dists
        # Tests if the comp_dists can call random with non None size
        with _DrawValuesContextBlocker():
            if isinstance(self.comp_dists, (list, tuple)):
                try:
                    [comp_dist.random(size=23)
                     for comp_dist in self.comp_dists]
                    self._comp_dists_vect = True
                except Exception:
                    # The comp_dists cannot call random with non None size or
                    # without knowledge of the point so we assume that we will
                    # have to iterate calls to random to get the correct size
                    self._comp_dists_vect = False
            else:
                try:
                    self.comp_dists.random(size=23)
                    self._comp_dists_vect = True
                except Exception:
                    # The comp_dists cannot call random with non None size or
                    # without knowledge of the point so we assume that we will
                    # have to iterate calls to random to get the correct size
                    self._comp_dists_vect = False

    def _comp_logp(self, value):
        comp_dists = self.comp_dists

        try:
            value_ = value if value.ndim > 1 else tt.shape_padright(value)

            return comp_dists.logp(value_)
        except AttributeError:
            return tt.squeeze(tt.stack([comp_dist.logp(value)
                                        for comp_dist in comp_dists],
                                       axis=1))

    def _comp_means(self):
        try:
            return tt.as_tensor_variable(self.comp_dists.mean)
        except AttributeError:
            return tt.squeeze(tt.stack([comp_dist.mean
                                        for comp_dist in self.comp_dists],
                                       axis=1))

    def _comp_modes(self):
        try:
            return tt.as_tensor_variable(self.comp_dists.mode)
        except AttributeError:
            return tt.squeeze(tt.stack([comp_dist.mode
                                        for comp_dist in self.comp_dists],
                                       axis=1))

    def _comp_samples(self, point=None, size=None):
        if self._comp_dists_vect or size is None:
            try:
                return self.comp_dists.random(point=point, size=size)
            except AttributeError:
                samples = np.array([comp_dist.random(point=point, size=size)
                                    for comp_dist in self.comp_dists])
                samples = np.moveaxis(samples, 0, samples.ndim - 1)
        else:
            # We must iterate the calls to random manually
            size = to_tuple(size)
            _size = int(np.prod(size))
            try:
                samples = np.array([self.comp_dists.random(point=point,
                                                           size=None)
                                    for _ in range(_size)])
                samples = np.reshape(samples, size + samples.shape[1:])
            except AttributeError:
                samples = np.array([[comp_dist.random(point=point, size=None)
                                     for _ in range(_size)]
                                    for comp_dist in self.comp_dists])
                samples = np.moveaxis(samples, 0, samples.ndim - 1)
                samples = np.reshape(samples, size + samples[1:])

        if samples.shape[-1] == 1:
            return samples[..., 0]
        else:
            return samples

    def logp(self, value):
        w = self.w

        return bound(logsumexp(tt.log(w) + self._comp_logp(value), axis=-1),
                     w >= 0, w <= 1, tt.allclose(w.sum(axis=-1), 1),
                     broadcast_conditions=False)

    def random(self, point=None, size=None):
        # Convert size to tuple
        size = to_tuple(size)
        # Draw mixture weights and a sample from each mixture to infer shape
        with _DrawValuesContext() as draw_context:
            # We first need to check w and comp_tmp shapes and re compute size
            w = draw_values([self.w], point=point, size=size)[0]
        with _DrawValuesContextBlocker():
            # We don't want to store the values drawn here in the context
            # because they wont have the correct size
            comp_tmp = self._comp_samples(point=point, size=None)

        # When size is not None, it's hard to tell the w parameter shape
        if size is not None and w.shape[:len(size)] == size:
            w_shape = w.shape[len(size):]
        else:
            w_shape = w.shape

        # Try to determine parameter shape and dist_shape
        param_shape = np.broadcast(np.empty(w_shape),
                                   comp_tmp).shape
        if np.asarray(self.shape).size != 0:
            dist_shape = np.broadcast(np.empty(self.shape),
                                      np.empty(param_shape[:-1])).shape
        else:
            dist_shape = param_shape[:-1]

        # When size is not None, maybe dist_shape partially overlaps with size
        if size is not None:
            if size == dist_shape:
                size = None
            elif size[-len(dist_shape):] == dist_shape:
                size = size[:len(size) - len(dist_shape)]

        # We get an integer _size instead of a tuple size for drawing the
        # mixture, then we just reshape the output
        if size is None:
            _size = None
        else:
            _size = int(np.prod(size))

        # Now we must broadcast w to the shape that considers size, dist_shape
        # and param_shape. However, we must take care with the cases in which
        # dist_shape and param_shape overlap
        if size is not None and w.shape[:len(size)] == size:
            if w.shape[:len(size + dist_shape)] != (size + dist_shape):
                # To allow w to broadcast, we insert new axis in between the
                # "size" axis and the "mixture" axis
                _w = w[(slice(None),) * len(size) +  # Index the size axis
                       (np.newaxis,) * len(dist_shape) +  # Add new axis for the dist_shape
                       (slice(None),)]  # Close with the slice of mixture components
                w = np.broadcast_to(_w, size + dist_shape + (param_shape[-1],))
        elif size is not None:
            w = np.broadcast_to(w, size + dist_shape + (param_shape[-1],))
        else:
            w = np.broadcast_to(w, dist_shape + (param_shape[-1],))

        # Compute the total size of the mixture's random call with size
        if _size is not None:
            output_size = int(_size * np.prod(dist_shape) * param_shape[-1])
        else:
            output_size = int(np.prod(dist_shape) * param_shape[-1])
        # Get the size we need for the mixture's random call
        mixture_size = int(output_size // np.prod(comp_tmp.shape))
        if mixture_size == 1 and _size is None:
            mixture_size = None

        # Semiflatten the mixture weights. The last axis is the number of
        # mixture mixture components, and the rest is all about size,
        # dist_shape and broadcasting
        w = np.reshape(w, (-1, w.shape[-1]))
        # Normalize mixture weights
        w = w / w.sum(axis=-1, keepdims=True)

        w_samples = generate_samples(random_choice,
                                     p=w,
                                     broadcast_shape=w.shape[:-1] or (1,),
                                     dist_shape=w.shape[:-1] or (1,),
                                     size=size)
        # Sample from the mixture
        with draw_context:
            mixed_samples = self._comp_samples(point=point,
                                               size=mixture_size)
        w_samples = w_samples.flatten()
        # Semiflatten the mixture to be able to zip it with w_samples
        mixed_samples = np.reshape(mixed_samples, (-1, comp_tmp.shape[-1]))
        # Select the samples from the mixture
        samples = np.array([mixed[choice] for choice, mixed in
                            zip(w_samples, mixed_samples)])
        # Reshape the samples to the correct output shape
        if size is None:
            samples = np.reshape(samples, dist_shape)
        else:
            samples = np.reshape(samples, size + dist_shape)
        return samples


class NormalMixture(Mixture):
    R"""
    Normal mixture log-likelihood

    .. math::

        f(x \mid w, \mu, \sigma^2) = \sum_{i = 1}^n w_i N(x \mid \mu_i, \sigma^2_i)

    ========  =======================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\sum_{i = 1}^n w_i \mu_i`
    Variance  :math:`\sum_{i = 1}^n w_i^2 \sigma^2_i`
    ========  =======================================

    Parameters
    ----------
    w : array of floats
        w >= 0 and w <= 1
        the mixture weights
    mu : array of floats
        the component means
    sigma : array of floats
        the component standard deviations
    tau : array of floats
        the component precisions
    comp_shape : shape of the Normal component
        notice that it should be different than the shape
        of the mixture distribution, with one axis being
        the number of components.

    Note: You only have to pass in sigma or tau, but not both.
    """

    def __init__(self, w, mu, comp_shape=(), *args, **kwargs):
        if 'sd' in kwargs.keys():
            kwargs['sigma'] = kwargs.pop('sd')

        _, sigma = get_tau_sigma(tau=kwargs.pop('tau', None),
                           sigma=kwargs.pop('sigma', None))

        self.mu = mu = tt.as_tensor_variable(mu)
        self.sigma = self.sd = sigma = tt.as_tensor_variable(sigma)

        super().__init__(w, Normal.dist(mu, sigma=sigma, shape=comp_shape),
                                            *args, **kwargs)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        w = dist.w
        sigma = dist.sigma
        name = r'\text{%s}' % name
        return r'${} \sim \text{{NormalMixture}}(\mathit{{w}}={},~\mathit{{mu}}={},~\mathit{{sigma}}={})$'.format(name,
                                                get_variable_name(w),
                                                get_variable_name(mu),
                                                get_variable_name(sigma))
