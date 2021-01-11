#   Copyright 2020 The PyMC Developers
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

from collections.abc import Iterable

import numpy as np
import theano
import theano.tensor as tt

from pymc3.distributions.continuous import Normal, get_tau_sigma
from pymc3.distributions.dist_math import bound, random_choice
from pymc3.distributions.distribution import (
    Discrete,
    Distribution,
    _DrawValuesContext,
    _DrawValuesContextBlocker,
    draw_values,
    generate_samples,
)
from pymc3.distributions.shape_utils import (
    broadcast_distribution_samples,
    get_broadcastable_dist_samples,
    to_tuple,
)
from pymc3.math import logsumexp
from pymc3.theanof import _conversion_map, take_along_axis

__all__ = ["Mixture", "NormalMixture", "MixtureSameFamily"]


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
    Support   :math:`\cup_{i = 1}^n \textrm{support}(f_i)`
    Mean      :math:`\sum_{i = 1}^n w_i \mu_i`
    ========  ============================================

    Parameters
    ----------
    w: array of floats
        w >= 0 and w <= 1
        the mixture weights
    comp_dists: multidimensional PyMC3 distribution (e.g. `pm.Poisson.dist(...)`)
        or iterable of PyMC3 distributions the component distributions
        :math:`f_1, \ldots, f_n`

    Examples
    --------
    .. code-block:: python

        # 2-Mixture Poisson distribution
        with pm.Model() as model:
            lam = pm.Exponential('lam', lam=1, shape=(2,))  # `shape=(2,)` indicates two mixture components.

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

        # npop-Mixture of multidimensional Gaussian
        npop = 5
        nd = (3, 4)
        with pm.Model() as model:
            mu = pm.Normal('mu', mu=np.arange(npop), sigma=1, shape=npop) # Each component has an independent mean

            w = pm.Dirichlet('w', a=np.ones(npop))

            components = pm.Normal.dist(mu=mu, sigma=1, shape=nd + (npop,))  # nd + (npop,) shaped multinomial

            like = pm.Mixture('like', w=w, comp_dists = components, observed=data, shape=nd)  # The resulting mixture is nd-shaped

        # Multidimensional Mixture as stacked independent mixtures
        with pm.Model() as model:
            mu = pm.Normal('mu', mu=np.arange(5), sigma=1, shape=5) # Each component has an independent mean

            w = pm.Dirichlet('w', a=np.ones(3, 5))  # w is a stack of 3 independent 5 component weight arrays

            components = pm.Normal.dist(mu=mu, sigma=1, shape=(3, 5))

            # The mixture is an array of 3 elements.
            # Each can be thought of as an independent scalar mixture of 5 components
            like = pm.Mixture('like', w=w, comp_dists = components, observed=data, shape=3)
    """

    def __init__(self, w, comp_dists, *args, **kwargs):
        # comp_dists type checking
        if not (
            isinstance(comp_dists, Distribution)
            or (
                isinstance(comp_dists, Iterable)
                and all(isinstance(c, Distribution) for c in comp_dists)
            )
        ):
            raise TypeError(
                "Supplied Mixture comp_dists must be a "
                "Distribution or an iterable of "
                "Distributions. Got {} instead.".format(
                    type(comp_dists)
                    if not isinstance(comp_dists, Iterable)
                    else [type(c) for c in comp_dists]
                )
            )
        shape = kwargs.pop("shape", ())

        self.w = w = tt.as_tensor_variable(w)
        self.comp_dists = comp_dists

        defaults = kwargs.pop("defaults", [])

        if all_discrete(comp_dists):
            default_dtype = _conversion_map[theano.config.floatX]
        else:
            default_dtype = theano.config.floatX

            try:
                self.mean = (w * self._comp_means()).sum(axis=-1)

                if "mean" not in defaults:
                    defaults.append("mean")
            except AttributeError:
                pass
        dtype = kwargs.pop("dtype", default_dtype)

        try:
            if isinstance(comp_dists, Distribution):
                comp_mode_logps = comp_dists.logp(comp_dists.mode)
            else:
                comp_mode_logps = tt.stack([cd.logp(cd.mode) for cd in comp_dists])

            mode_idx = tt.argmax(tt.log(w) + comp_mode_logps, axis=-1)
            self.mode = self._comp_modes()[mode_idx]

            if "mode" not in defaults:
                defaults.append("mode")
        except (AttributeError, ValueError, IndexError):
            pass

        super().__init__(shape, dtype, defaults=defaults, *args, **kwargs)

    @property
    def comp_dists(self):
        return self._comp_dists

    @comp_dists.setter
    def comp_dists(self, comp_dists):
        self._comp_dists = comp_dists
        if isinstance(comp_dists, Distribution):
            self._comp_dist_shapes = to_tuple(comp_dists.shape)
            self._broadcast_shape = self._comp_dist_shapes
            self.comp_is_distribution = True
        else:
            # Now we check the comp_dists distribution shape, see what
            # the broadcast shape would be. This shape will be the dist_shape
            # used by generate samples (the shape of a single random sample)
            # from the mixture
            self._comp_dist_shapes = [to_tuple(d.shape) for d in comp_dists]
            # All component distributions must broadcast with each other
            try:
                self._broadcast_shape = np.broadcast(
                    *[np.empty(shape) for shape in self._comp_dist_shapes]
                ).shape
            except Exception:
                raise TypeError(
                    "Supplied comp_dists shapes do not broadcast "
                    "with each other. comp_dists shapes are: "
                    "{}".format(self._comp_dist_shapes)
                )

            # We wrap the _comp_dist.random by adding the kwarg raw_size_,
            # which will be the size attribute passed to _comp_samples.
            # _comp_samples then calls generate_samples, which may change the
            # size value to make it compatible with scipy.stats.*.rvs
            self._generators = []
            for comp_dist in comp_dists:
                generator = Mixture._comp_dist_random_wrapper(comp_dist.random)
                self._generators.append(generator)
            self.comp_is_distribution = False

    @staticmethod
    def _comp_dist_random_wrapper(random):
        """Wrap the comp_dists.random method to take the kwarg raw_size_ and
        use it's value to replace the size parameter. This is needed because
        generate_samples makes the size value compatible with the
        scipy.stats.*.rvs, where size has a different meaning than in the
        distributions' random methods.
        """

        def wrapped_random(*args, **kwargs):
            raw_size_ = kwargs.pop("raw_size_", None)
            # Distribution.random's signature is always (point=None, size=None)
            # so size could be the second arg or be given as a kwarg
            if len(args) > 1:
                args[1] = raw_size_
            else:
                kwargs["size"] = raw_size_
            return random(*args, **kwargs)

        return wrapped_random

    def _comp_logp(self, value):
        comp_dists = self.comp_dists

        if self.comp_is_distribution:
            # Value can be many things. It can be the self tensor, the mode
            # test point or it can be observed data. The latter case requires
            # careful handling of shape, as the observed's shape could look
            # like (repetitions,) + dist_shape, which does not include the last
            # mixture axis. For this reason, we try to eval the value.shape,
            # compare it with self.shape and shape_padright if we infer that
            # the value holds observed data
            try:
                val_shape = tuple(value.shape.eval())
            except AttributeError:
                val_shape = value.shape
            except theano.graph.fg.MissingInputError:
                val_shape = None
            try:
                self_shape = tuple(self.shape)
            except AttributeError:
                # Happens in __init__ when computing self.logp(comp_modes)
                self_shape = None
            comp_shape = tuple(comp_dists.shape)
            ndim = value.ndim
            if val_shape is not None and not (
                (self_shape is not None and val_shape == self_shape) or val_shape == comp_shape
            ):
                # value is neither the test point nor the self tensor, it
                # is likely to hold observed values, so we must compute the
                # ndim discarding the dimensions that don't match
                # self_shape
                if self_shape and val_shape[-len(self_shape) :] == self_shape:
                    # value has observed values for the Mixture
                    ndim = len(self_shape)
                elif comp_shape and val_shape[-len(comp_shape) :] == comp_shape:
                    # value has observed for the Mixture components
                    ndim = len(comp_shape)
                else:
                    # We cannot infer what was passed, we handle this
                    # as was done in earlier versions of Mixture. We pad
                    # always if ndim is lower or equal to 1  (default
                    # legacy implementation)
                    if ndim <= 1:
                        ndim = len(comp_dists.shape) - 1
            else:
                # We reach this point if value does not hold observed data, so
                # we can use its ndim safely to determine shape padding, or it
                # holds something that we cannot infer, so we revert to using
                # the value's ndim for shape padding.
                # We will always pad a single dimension if ndim is lower or
                # equal to 1 (default legacy implementation)
                if ndim <= 1:
                    ndim = len(comp_dists.shape) - 1
            if ndim < len(comp_dists.shape):
                value_ = tt.shape_padright(value, len(comp_dists.shape) - ndim)
            else:
                value_ = value
            return comp_dists.logp(value_)
        else:
            return tt.squeeze(
                tt.stack([comp_dist.logp(value) for comp_dist in comp_dists], axis=-1)
            )

    def _comp_means(self):
        try:
            return tt.as_tensor_variable(self.comp_dists.mean)
        except AttributeError:
            return tt.squeeze(tt.stack([comp_dist.mean for comp_dist in self.comp_dists], axis=-1))

    def _comp_modes(self):
        try:
            return tt.as_tensor_variable(self.comp_dists.mode)
        except AttributeError:
            return tt.squeeze(tt.stack([comp_dist.mode for comp_dist in self.comp_dists], axis=-1))

    def _comp_samples(self, point=None, size=None, comp_dist_shapes=None, broadcast_shape=None):
        if self.comp_is_distribution:
            samples = self._comp_dists.random(point=point, size=size)
        else:
            if comp_dist_shapes is None:
                comp_dist_shapes = self._comp_dist_shapes
            if broadcast_shape is None:
                broadcast_shape = self._sample_shape
            samples = []
            for dist_shape, generator in zip(comp_dist_shapes, self._generators):
                sample = generate_samples(
                    generator=generator,
                    dist_shape=dist_shape,
                    broadcast_shape=broadcast_shape,
                    point=point,
                    size=size,
                    not_broadcast_kwargs={"raw_size_": size},
                )
                samples.append(sample)
            samples = np.array(broadcast_distribution_samples(samples, size=size))
            # In the logp we assume the last axis holds the mixture components
            # so we move the axis to the last dimension
            samples = np.moveaxis(samples, 0, -1)
        return samples.astype(self.dtype)

    def infer_comp_dist_shapes(self, point=None):
        """Try to infer the shapes of the component distributions,
        `comp_dists`, and how they should broadcast together.
        The behavior is slightly different if `comp_dists` is a `Distribution`
        as compared to when it is a list of `Distribution`s. When it is a list
        the following procedure is repeated for each element in the list:
        1. Look up the `comp_dists.shape`
        2. If it is not empty, use it as `comp_dist_shape`
        3. If it is an empty tuple, a single random sample is drawn by calling
        `comp_dists.random(point=point, size=None)`, and the returned
        test_sample's shape is used as the inferred `comp_dists.shape`

        Parameters
        ----------
        point: None or dict (optional)
            Dictionary that maps rv names to values, to supply to
            `self.comp_dists.random`

        Returns
        -------
        comp_dist_shapes: shape tuple or list of shape tuples.
            If `comp_dists` is a `Distribution`, it is a shape tuple of the
            inferred distribution shape.
            If `comp_dists` is a list of `Distribution`s, it is a list of
            shape tuples inferred for each element in `comp_dists`
        broadcast_shape: shape tuple
            The shape that results from broadcasting all component's shapes
            together.
        """
        if self.comp_is_distribution:
            if len(self._comp_dist_shapes) > 0:
                comp_dist_shapes = self._comp_dist_shapes
            else:
                # Happens when the distribution is a scalar or when it was not
                # given a shape. In these cases we try to draw a single value
                # to check its shape, we use the provided point dictionary
                # hoping that it can circumvent the Flat and HalfFlat
                # undrawable distributions.
                with _DrawValuesContextBlocker():
                    test_sample = self._comp_dists.random(point=point, size=None)
                    comp_dist_shapes = test_sample.shape
            broadcast_shape = comp_dist_shapes
        else:
            # Now we check the comp_dists distribution shape, see what
            # the broadcast shape would be. This shape will be the dist_shape
            # used by generate samples (the shape of a single random sample)
            # from the mixture
            comp_dist_shapes = []
            for dist_shape, comp_dist in zip(self._comp_dist_shapes, self._comp_dists):
                if dist_shape == tuple():
                    # Happens when the distribution is a scalar or when it was
                    # not given a shape. In these cases we try to draw a single
                    # value to check its shape, we use the provided point
                    # dictionary hoping that it can circumvent the Flat and
                    # HalfFlat undrawable distributions.
                    with _DrawValuesContextBlocker():
                        test_sample = comp_dist.random(point=point, size=None)
                        dist_shape = test_sample.shape
                comp_dist_shapes.append(dist_shape)
            # All component distributions must broadcast with each other
            try:
                broadcast_shape = np.broadcast(
                    *[np.empty(shape) for shape in comp_dist_shapes]
                ).shape
            except Exception:
                raise TypeError(
                    "Inferred comp_dist shapes do not broadcast "
                    "with each other. comp_dists inferred shapes "
                    "are: {}".format(comp_dist_shapes)
                )
        return comp_dist_shapes, broadcast_shape

    def logp(self, value):
        """
        Calculate log-probability of defined Mixture distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        w = self.w

        return bound(
            logsumexp(tt.log(w) + self._comp_logp(value), axis=-1, keepdims=False),
            w >= 0,
            w <= 1,
            tt.allclose(w.sum(axis=-1), 1),
            broadcast_conditions=False,
        )

    def random(self, point=None, size=None):
        """
        Draw random values from defined Mixture distribution.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        # Convert size to tuple
        size = to_tuple(size)
        # Draw mixture weights and infer the comp_dists shapes
        with _DrawValuesContext() as draw_context:
            # We first need to check w and comp_tmp shapes and re compute size
            w = draw_values([self.w], point=point, size=size)[0]
            comp_dist_shapes, broadcast_shape = self.infer_comp_dist_shapes(point=point)

        # When size is not None, it's hard to tell the w parameter shape
        if size is not None and w.shape[: len(size)] == size:
            w_shape = w.shape[len(size) :]
        else:
            w_shape = w.shape

        # Try to determine parameter shape and dist_shape
        if self.comp_is_distribution:
            param_shape = np.broadcast(np.empty(w_shape), np.empty(broadcast_shape)).shape
        else:
            param_shape = np.broadcast(np.empty(w_shape), np.empty(broadcast_shape + (1,))).shape
        if np.asarray(self.shape).size != 0:
            dist_shape = np.broadcast(np.empty(self.shape), np.empty(param_shape[:-1])).shape
        else:
            dist_shape = param_shape[:-1]

        # Try to determine the size that must be used to get the mixture
        # components (i.e. get random choices using w).
        # 1. There must be size independent choices based on w.
        # 2. There must also be independent draws for each non singleton axis
        # of w.
        # 3. There must also be independent draws for each dimension added by
        # self.shape with respect to the w.ndim. These usually correspond to
        # observed variables with batch shapes
        wsh = (1,) * (len(dist_shape) - len(w_shape) + 1) + w_shape[:-1]
        psh = (1,) * (len(dist_shape) - len(param_shape) + 1) + param_shape[:-1]
        w_sample_size = []
        # Loop through the dist_shape to get the conditions 2 and 3 first
        for i in range(len(dist_shape)):
            if dist_shape[i] != psh[i] and wsh[i] == 1:
                # self.shape[i] is a non singleton dimension (usually caused by
                # observed data)
                sh = dist_shape[i]
            else:
                sh = wsh[i]
            w_sample_size.append(sh)
        if size is not None and w_sample_size[: len(size)] != size:
            w_sample_size = size + tuple(w_sample_size)
        # Broadcast w to the w_sample_size (add a singleton last axis for the
        # mixture components)
        w = broadcast_distribution_samples([w, np.empty(w_sample_size + (1,))], size=size)[0]

        # Semiflatten the mixture weights. The last axis is the number of
        # mixture mixture components, and the rest is all about size,
        # dist_shape and broadcasting
        w_ = np.reshape(w, (-1, w.shape[-1]))
        w_samples = random_choice(p=w_, size=None)  # w's shape already includes size
        # Now we broadcast the chosen components to the dist_shape
        w_samples = np.reshape(w_samples, w.shape[:-1])
        if size is not None and dist_shape[: len(size)] != size:
            w_samples = np.broadcast_to(w_samples, size + dist_shape)
        else:
            w_samples = np.broadcast_to(w_samples, dist_shape)

        # When size is not None, maybe dist_shape partially overlaps with size
        if size is not None:
            if size == dist_shape:
                size = None
            elif size[-len(dist_shape) :] == dist_shape:
                size = size[: len(size) - len(dist_shape)]

        # We get an integer _size instead of a tuple size for drawing the
        # mixture, then we just reshape the output
        if size is None:
            _size = None
        else:
            _size = int(np.prod(size))

        # Compute the total size of the mixture's random call with size
        if _size is not None:
            output_size = int(_size * np.prod(dist_shape) * param_shape[-1])
        else:
            output_size = int(np.prod(dist_shape) * param_shape[-1])
        # Get the size we need for the mixture's random call
        if self.comp_is_distribution:
            mixture_size = int(output_size // np.prod(broadcast_shape))
        else:
            mixture_size = int(output_size // (np.prod(broadcast_shape) * param_shape[-1]))
        if mixture_size == 1 and _size is None:
            mixture_size = None

        # Sample from the mixture
        with draw_context:
            mixed_samples = self._comp_samples(
                point=point,
                size=mixture_size,
                broadcast_shape=broadcast_shape,
                comp_dist_shapes=comp_dist_shapes,
            )
        # Test that the mixture has the same number of "samples" as w
        if w_samples.size != (mixed_samples.size // w.shape[-1]):
            raise ValueError(
                "Inconsistent number of samples from the "
                "mixture and mixture weights. Drew {} mixture "
                "weights elements, and {} samples from the "
                "mixture components.".format(w_samples.size, mixed_samples.size // w.shape[-1])
            )
        # Semiflatten the mixture to be able to zip it with w_samples
        w_samples = w_samples.flatten()
        mixed_samples = np.reshape(mixed_samples, (-1, w.shape[-1]))
        # Select the samples from the mixture
        samples = np.array([mixed[choice] for choice, mixed in zip(w_samples, mixed_samples)])
        # Reshape the samples to the correct output shape
        if size is None:
            samples = np.reshape(samples, dist_shape)
        else:
            samples = np.reshape(samples, size + dist_shape)
        return samples

    def _distr_parameters_for_repr(self):
        return []


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
    w: array of floats
        w >= 0 and w <= 1
        the mixture weights
    mu: array of floats
        the component means
    sigma: array of floats
        the component standard deviations
    tau: array of floats
        the component precisions
    comp_shape: shape of the Normal component
        notice that it should be different than the shape
        of the mixture distribution, with one axis being
        the number of components.

    Notes
    -----
    You only have to pass in sigma or tau, but not both.

    Examples
    --------
    .. code-block:: python

        n_components = 3

        with pm.Model() as gauss_mix:
            μ = pm.Normal(
                "μ",
                data.mean(),
                10,
                shape=n_components,
                transform=pm.transforms.ordered,
                testval=[1, 2, 3],
            )
            σ = pm.HalfNormal("σ", 10, shape=n_components)
            weights = pm.Dirichlet("w", np.ones(n_components))

            pm.NormalMixture("y", w=weights, mu=μ, sigma=σ, observed=data)
    """

    def __init__(self, w, mu, sigma=None, tau=None, sd=None, comp_shape=(), *args, **kwargs):
        if sd is not None:
            sigma = sd
        _, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        self.mu = mu = tt.as_tensor_variable(mu)
        self.sigma = self.sd = sigma = tt.as_tensor_variable(sigma)

        super().__init__(w, Normal.dist(mu, sigma=sigma, shape=comp_shape), *args, **kwargs)

    def _distr_parameters_for_repr(self):
        return ["w", "mu", "sigma"]


class MixtureSameFamily(Distribution):
    R"""
    Mixture Same Family log-likelihood
    This distribution handles mixtures of multivariate distributions in a vectorized
    manner. It is used over Mixture distribution when the mixture components are not
    present on the last axis of components' distribution.

    .. math::f(x \mid w, \theta) = \sum_{i = 1}^n w_i f_i(x \mid \theta_i)\textrm{ Along mixture\_axis}

    ========  ============================================
    Support   :math:`\textrm{support}(f)`
    Mean      :math:`w\mu`
    ========  ============================================

    Parameters
    ----------
    w: array of floats
        w >= 0 and w <= 1
        the mixture weights
    comp_dists: PyMC3 distribution (e.g. `pm.Multinomial.dist(...)`)
        The `comp_dists` can be scalar or multidimensional distribution.
        Assuming its shape to be - (i_0, ..., i_n, mixture_axis, i_n+1, ..., i_N),
        the `mixture_axis` is consumed resulting in the shape of mixture as -
        (i_0, ..., i_n, i_n+1, ..., i_N).
    mixture_axis: int, default = -1
        Axis representing the mixture components to be reduced in the mixture.

    Notes
    -----
    The default behaviour resembles Mixture distribution wherein the last axis of component
    distribution is reduced.
    """

    def __init__(self, w, comp_dists, mixture_axis=-1, *args, **kwargs):
        self.w = tt.as_tensor_variable(w)
        if not isinstance(comp_dists, Distribution):
            raise TypeError(
                "The MixtureSameFamily distribution only accepts Distribution "
                f"instances as its components. Got {type(comp_dists)} instead."
            )
        self.comp_dists = comp_dists
        if mixture_axis < 0:
            mixture_axis = len(comp_dists.shape) + mixture_axis
            if mixture_axis < 0:
                raise ValueError(
                    "`mixture_axis` is supposed to be in shape of components' distribution. "
                    f"Got {mixture_axis + len(comp_dists.shape)} axis instead out of the bounds."
                )
        comp_shape = to_tuple(comp_dists.shape)
        self.shape = comp_shape[:mixture_axis] + comp_shape[mixture_axis + 1 :]
        self.mixture_axis = mixture_axis
        kwargs.setdefault("dtype", self.comp_dists.dtype)

        # Compute the mode so we don't always have to pass a testval
        defaults = kwargs.pop("defaults", [])
        event_shape = self.comp_dists.shape[mixture_axis + 1 :]
        _w = tt.shape_padleft(
            tt.shape_padright(w, len(event_shape)),
            len(self.comp_dists.shape) - w.ndim - len(event_shape),
        )
        mode = take_along_axis(
            self.comp_dists.mode,
            tt.argmax(_w, keepdims=True),
            axis=mixture_axis,
        )
        self.mode = mode[(..., 0) + (slice(None),) * len(event_shape)]

        if not all_discrete(comp_dists):
            mean = tt.as_tensor_variable(self.comp_dists.mean)
            self.mean = (_w * mean).sum(axis=mixture_axis)
            if "mean" not in defaults:
                defaults.append("mean")
        defaults.append("mode")

        super().__init__(defaults=defaults, *args, **kwargs)

    def logp(self, value):
        """
        Calculate log-probability of defined ``MixtureSameFamily`` distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """

        comp_dists = self.comp_dists
        w = self.w
        mixture_axis = self.mixture_axis

        event_shape = comp_dists.shape[mixture_axis + 1 :]

        # To be able to broadcast the comp_dists.logp with w and value
        # We first have to pad the shape of w to the right with ones
        # so that it can broadcast with the event_shape.

        w = tt.shape_padright(w, len(event_shape))

        # Second, we have to add the mixture_axis to the value tensor
        # To insert the mixture axis at the correct location, we use the
        # negative number index. This way, we can also handle situations
        # in which, value is an observed value with more batch dimensions
        # than the ones present in the comp_dists.
        comp_dists_ndim = len(comp_dists.shape)

        value = tt.shape_padaxis(value, axis=mixture_axis - comp_dists_ndim)

        comp_logp = comp_dists.logp(value)
        return bound(
            logsumexp(tt.log(w) + comp_logp, axis=mixture_axis, keepdims=False),
            w >= 0,
            w <= 1,
            tt.allclose(w.sum(axis=mixture_axis - comp_dists_ndim), 1),
            broadcast_conditions=False,
        )

    def random(self, point=None, size=None):
        """
        Draw random values from defined ``MixtureSameFamily`` distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        sample_shape = to_tuple(size)
        mixture_axis = self.mixture_axis

        # First we draw values for the mixture component weights
        (w,) = draw_values([self.w], point=point, size=size)

        # We now draw random choices from those weights.
        # However, we have to ensure that the number of choices has the
        # sample_shape present.
        w_shape = w.shape
        batch_shape = self.comp_dists.shape[: mixture_axis + 1]
        param_shape = np.broadcast(np.empty(w_shape), np.empty(batch_shape)).shape
        event_shape = self.comp_dists.shape[mixture_axis + 1 :]

        if np.asarray(self.shape).size != 0:
            comp_dists_ndim = len(self.comp_dists.shape)

            # If event_shape of both comp_dists and supplied shape matches,
            # broadcast only batch_shape
            # else broadcast the entire given shape with batch_shape.
            if list(self.shape[mixture_axis - comp_dists_ndim + 1 :]) == list(event_shape):
                dist_shape = np.broadcast(
                    np.empty(self.shape[:mixture_axis]), np.empty(param_shape[:mixture_axis])
                ).shape
            else:
                dist_shape = np.broadcast(
                    np.empty(self.shape), np.empty(param_shape[:mixture_axis])
                ).shape
        else:
            dist_shape = param_shape[:mixture_axis]

        # Try to determine the size that must be used to get the mixture
        # components (i.e. get random choices using w).
        # 1. There must be size independent choices based on w.
        # 2. There must also be independent draws for each non singleton axis
        # of w.
        # 3. There must also be independent draws for each dimension added by
        # self.shape with respect to the w.ndim. These usually correspond to
        # observed variables with batch shapes
        wsh = (1,) * (len(dist_shape) - len(w_shape) + 1) + w_shape[:mixture_axis]
        psh = (1,) * (len(dist_shape) - len(param_shape) + 1) + param_shape[:mixture_axis]
        w_sample_size = []
        # Loop through the dist_shape to get the conditions 2 and 3 first
        for i in range(len(dist_shape)):
            if dist_shape[i] != psh[i] and wsh[i] == 1:
                # self.shape[i] is a non singleton dimension (usually caused by
                # observed data)
                sh = dist_shape[i]
            else:
                sh = wsh[i]
            w_sample_size.append(sh)

        if sample_shape is not None and w_sample_size[: len(sample_shape)] != sample_shape:
            w_sample_size = sample_shape + tuple(w_sample_size)

        choices = random_choice(p=w, size=w_sample_size)

        # We now draw samples from the mixture components random method
        comp_samples = self.comp_dists.random(point=point, size=size)
        if comp_samples.shape[: len(sample_shape)] != sample_shape:
            comp_samples = np.broadcast_to(
                comp_samples,
                shape=sample_shape + comp_samples.shape,
            )

        # At this point the shapes of the arrays involved are:
        # comp_samples.shape = (sample_shape, batch_shape, mixture_axis, event_shape)
        # choices.shape = (sample_shape, batch_shape)
        #
        # To be able to take the choices along the mixture_axis of the
        # comp_samples, we have to add in dimensions to the right of the
        # choices array.
        # We also need to make sure that the batch_shapes of both the comp_samples
        # and choices broadcast with each other.

        choices = np.reshape(choices, choices.shape + (1,) * (1 + len(event_shape)))

        choices, comp_samples = get_broadcastable_dist_samples([choices, comp_samples], size=size)

        # We now take the choices of the mixture components along the mixture_axis
        # but we use the negative index representation to be able to handle the
        # sample_shape
        samples = np.take_along_axis(
            comp_samples, choices, axis=mixture_axis - len(self.comp_dists.shape)
        )

        # The `samples` array still has the `mixture_axis`, so we must remove it:
        output = samples[(..., 0) + (slice(None),) * len(event_shape)]
        return output

    def _distr_parameters_for_repr(self):
        return []
