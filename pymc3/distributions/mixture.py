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
from typing import List, Optional, Union

import aesara
import aesara.tensor as at
import numpy as np

from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable

from pymc3.aesaraf import _conversion_map, take_along_axis
from pymc3.distributions.continuous import Normal, get_tau_sigma
from pymc3.distributions.dist_math import bound
from pymc3.distributions.distribution import Discrete, Distribution
from pymc3.distributions.shape_utils import to_tuple
from pymc3.math import logsumexp

__all__ = ["Mixture", "NormalMixture", "MixtureSameFamily"]


def all_discrete(comp_dists):
    """
    Determine if all distributions in comp_dists are discrete
    """
    if isinstance(comp_dists, Distribution):
        return isinstance(comp_dists, Discrete)
    else:
        return all(isinstance(comp_dist, Discrete) for comp_dist in comp_dists)


class MixtureRV(RandomVariable):
    name = "mixture"
    ndim_supp = 0
    ndims_params = [1, 1]
    _print_name = ("Mixture", "\\operatorname{Mixture")

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        w: Union[np.ndarray, float],
        comp_dist: Union[Distribution, Iterable[Distribution]],
        size: Optional[Union[List[int], int]] = None,
    ) -> np.ndarray:

        component = rng.multinomial(n=1, pvals=w)

        return comp_dist[component].rv_op.rng_fn(rng)


mixture = MixtureRV()


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
    rv_op = mixture

    @classmethod
    def dist(cls, w, comp_dists, *args, **kwargs):
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

        w = at.as_tensor_variable(w)

        defaults = kwargs.pop("defaults", [])

        if all_discrete(comp_dists):
            default_dtype = _conversion_map[aesara.config.floatX]
        else:
            default_dtype = aesara.config.floatX

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
                comp_mode_logps = at.stack([cd.logp(cd.mode) for cd in comp_dists])

            mode_idx = at.argmax(at.log(w) + comp_mode_logps, axis=-1)
            self.mode = self._comp_modes()[mode_idx]

            if "mode" not in defaults:
                defaults.append("mode")
        except (AttributeError, ValueError, IndexError):
            pass

        return super().dist([w, comp_dists], *args, **kwargs)

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
                    *(np.empty(shape) for shape in self._comp_dist_shapes)
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
            except aesara.graph.fg.MissingInputError:
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
                value_ = at.shape_padright(value, len(comp_dists.shape) - ndim)
            else:
                value_ = value
            return comp_dists.logp(value_)
        else:
            return at.squeeze(
                at.stack([comp_dist.logp(value) for comp_dist in comp_dists], axis=-1)
            )

    def _comp_means(self):
        try:
            return at.as_tensor_variable(self.comp_dists.mean)
        except AttributeError:
            return at.squeeze(at.stack([comp_dist.mean for comp_dist in self.comp_dists], axis=-1))

    def _comp_modes(self):
        try:
            return at.as_tensor_variable(self.comp_dists.mode)
        except AttributeError:
            return at.squeeze(at.stack([comp_dist.mode for comp_dist in self.comp_dists], axis=-1))

    def _comp_samples(self, point=None, size=None, comp_dist_shapes=None, broadcast_shape=None):
        # if self.comp_is_distribution:
        #     samples = self._comp_dists.random(point=point, size=size)
        # else:
        #     if comp_dist_shapes is None:
        #         comp_dist_shapes = self._comp_dist_shapes
        #     if broadcast_shape is None:
        #         broadcast_shape = self._sample_shape
        #     samples = []
        #     for dist_shape, generator in zip(comp_dist_shapes, self._generators):
        #         sample = generate_samples(
        #             generator=generator,
        #             dist_shape=dist_shape,
        #             broadcast_shape=broadcast_shape,
        #             point=point,
        #             size=size,
        #             not_broadcast_kwargs={"raw_size_": size},
        #         )
        #         samples.append(sample)
        #     samples = np.array(broadcast_distribution_samples(samples, size=size))
        #     # In the logp we assume the last axis holds the mixture components
        #     # so we move the axis to the last dimension
        #     samples = np.moveaxis(samples, 0, -1)
        # return samples.astype(self.dtype)
        pass

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
        # if self.comp_is_distribution:
        #     if len(self._comp_dist_shapes) > 0:
        #         comp_dist_shapes = self._comp_dist_shapes
        #     else:
        #         # Happens when the distribution is a scalar or when it was not
        #         # given a shape. In these cases we try to draw a single value
        #         # to check its shape, we use the provided point dictionary
        #         # hoping that it can circumvent the Flat and HalfFlat
        #         # undrawable distributions.
        #         with _DrawValuesContextBlocker():
        #             test_sample = self._comp_dists.random(point=point, size=None)
        #             comp_dist_shapes = test_sample.shape
        #     broadcast_shape = comp_dist_shapes
        # else:
        #     # Now we check the comp_dists distribution shape, see what
        #     # the broadcast shape would be. This shape will be the dist_shape
        #     # used by generate samples (the shape of a single random sample)
        #     # from the mixture
        #     comp_dist_shapes = []
        #     for dist_shape, comp_dist in zip(self._comp_dist_shapes, self._comp_dists):
        #         if dist_shape == tuple():
        #             # Happens when the distribution is a scalar or when it was
        #             # not given a shape. In these cases we try to draw a single
        #             # value to check its shape, we use the provided point
        #             # dictionary hoping that it can circumvent the Flat and
        #             # HalfFlat undrawable distributions.
        #             with _DrawValuesContextBlocker():
        #                 test_sample = comp_dist.random(point=point, size=None)
        #                 dist_shape = test_sample.shape
        #         comp_dist_shapes.append(dist_shape)
        #     # All component distributions must broadcast with each other
        #     try:
        #         broadcast_shape = np.broadcast(
        #             *[np.empty(shape) for shape in comp_dist_shapes]
        #         ).shape
        #     except Exception:
        #         raise TypeError(
        #             "Inferred comp_dist shapes do not broadcast "
        #             "with each other. comp_dists inferred shapes "
        #             "are: {}".format(comp_dist_shapes)
        #         )
        # return comp_dist_shapes, broadcast_shape

    def logp(value: Union[float, np.ndarray, TensorVariable]):
        """
        Calculate log-probability of defined Mixture distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        w = self.w

        return bound(
            logsumexp(at.log(w) + self._comp_logp(value), axis=-1, keepdims=False),
            w >= 0,
            w <= 1,
            at.allclose(w.sum(axis=-1), 1),
            broadcast_conditions=False,
        )

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
                initval=[1, 2, 3],
            )
            σ = pm.HalfNormal("σ", 10, shape=n_components)
            weights = pm.Dirichlet("w", np.ones(n_components))

            pm.NormalMixture("y", w=weights, mu=μ, sigma=σ, observed=data)
    """

    @classmethod
    def dist(cls, w, mu, sigma=None, tau=None, sd=None, comp_shape=(), *args, **kwargs):
        if sd is not None:
            sigma = sd
        _, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        self.mu = mu = at.as_tensor_variable(mu)
        self.sigma = self.sd = sigma = at.as_tensor_variable(sigma)

        super().dist([w, Normal.dist(mu, sigma=sigma, shape=comp_shape)], *args, **kwargs)

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

    @classmethod
    def __dist__(w, comp_dists, mixture_axis=-1, *args, **kwargs):
        self.w = at.as_tensor_variable(w)
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

        # Compute the mode so we don't always have to pass a initval
        defaults = kwargs.pop("defaults", [])
        event_shape = self.comp_dists.shape[mixture_axis + 1 :]
        _w = at.shape_padleft(
            at.shape_padright(w, len(event_shape)),
            len(self.comp_dists.shape) - w.ndim - len(event_shape),
        )
        mode = take_along_axis(
            self.comp_dists.mode,
            at.argmax(_w, keepdims=True),
            axis=mixture_axis,
        )
        self.mode = mode[(..., 0) + (slice(None),) * len(event_shape)]

        if not all_discrete(comp_dists):
            mean = at.as_tensor_variable(self.comp_dists.mean)
            self.mean = (_w * mean).sum(axis=mixture_axis)
            if "mean" not in defaults:
                defaults.append("mean")
        defaults.append("mode")

        super().dist([w, comp_dists], defaults=defaults, *args, **kwargs)

    def logp(value):
        """
        Calculate log-probability of defined ``MixtureSameFamily`` distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

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

        w = at.shape_padright(w, len(event_shape))

        # Second, we have to add the mixture_axis to the value tensor
        # To insert the mixture axis at the correct location, we use the
        # negative number index. This way, we can also handle situations
        # in which, value is an observed value with more batch dimensions
        # than the ones present in the comp_dists.
        comp_dists_ndim = len(comp_dists.shape)

        value = at.shape_padaxis(value, axis=mixture_axis - comp_dists_ndim)

        comp_logp = comp_dists.logp(value)
        return bound(
            logsumexp(at.log(w) + comp_logp, axis=mixture_axis, keepdims=False),
            w >= 0,
            w <= 1,
            at.allclose(w.sum(axis=mixture_axis - comp_dists_ndim), 1),
            broadcast_conditions=False,
        )

    def _distr_parameters_for_repr(self):
        return []
