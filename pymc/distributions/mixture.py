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
import aesara
import aesara.tensor as at
import numpy as np

from aeppl.abstract import MeasurableVariable, _get_measurable_outputs
from aeppl.logprob import _logprob
from aesara.compile.builders import OpFromGraph
from aesara.tensor import TensorVariable
from aesara.tensor.random.op import RandomVariable

from pymc.aesaraf import change_rv_size, take_along_axis
from pymc.distributions.continuous import Normal
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Discrete, Distribution, SymbolicDistribution
from pymc.distributions.logprob import logp
from pymc.distributions.shape_utils import to_tuple
from pymc.math import logsumexp
from pymc.util import check_dist_not_registered

__all__ = ["Mixture", "NormalMixture", "MixtureSameFamily"]


def all_discrete(comp_dists):
    """
    Determine if all distributions in comp_dists are discrete
    """
    if isinstance(comp_dists, Distribution):
        return isinstance(comp_dists, Discrete)
    else:
        return all(isinstance(comp_dist, Discrete) for comp_dist in comp_dists)


class MarginalMixtureRV(OpFromGraph):
    """A placeholder used to specify a log-likelihood for a mixture sub-graph."""


MeasurableVariable.register(MarginalMixtureRV)


class Mixture(SymbolicDistribution):
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
    comp_dists: multidimensional PyMC distribution (e.g. `pm.Poisson.dist(...)`)
        or iterable of PyMC distributions the component distributions
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

    @classmethod
    def dist(cls, w, comp_dists, **kwargs):
        if not isinstance(comp_dists, (tuple, list)):
            # comp_dists is a single component
            comp_dists = [comp_dists]

        # Check that components are not associated with a registered variable in the model
        components_ndim = set()
        components_ndim_supp = set()
        for dist in comp_dists:
            # TODO: Allow these to not be a RandomVariable as long as we can call `ndim_supp` on them
            #  and resize them
            if not isinstance(dist, TensorVariable) or not isinstance(
                dist.owner.op, RandomVariable
            ):
                raise ValueError(
                    f"Component dist must be a distribution created via the `.dist()` API, got {type(dist)}"
                )
            check_dist_not_registered(dist)
            components_ndim.add(dist.ndim)
            components_ndim_supp.add(dist.owner.op.ndim_supp)

        if len(components_ndim) > 1:
            raise ValueError(
                f"Mixture components must all have the same dimensionality, got {components_ndim}"
            )

        if len(components_ndim_supp) > 1:
            raise ValueError(
                f"Mixture components must all have the same support dimensionality, got {components_ndim_supp}"
            )

        w = at.as_tensor_variable(w)
        return super().dist([w, *comp_dists], **kwargs)

    @classmethod
    def rv_op(cls, weights, *components, size=None, rngs=None):
        # Update rngs if provided
        if rngs is not None:
            components = cls._reseed_components(rngs, *components)
            *_, mix_indexes_rng = rngs
        else:
            # Create new rng for the mix_indexes internal RV
            mix_indexes_rng = aesara.shared(np.random.default_rng())

        if size is not None:
            components = cls._resize_components(size, *components)

        single_component = len(components) == 1

        # Extract support and replication ndims from components and weights
        ndim_supp = components[0].owner.op.ndim_supp
        ndim_batch = components[0].ndim - ndim_supp
        if single_component:
            # One dimension is taken by the mixture axis in the single component case
            ndim_batch -= 1

        # The weights may imply extra batch dimensions that go beyond what is already
        # implied by the component dimensions (ndim_batch)
        weights_ndim_batch = max(0, weights.ndim - ndim_batch - 1)

        # If weights are large enough that they would broadcast the component distributions
        # we try to resize them. This in necessary to avoid duplicated values in the
        # random method and for equivalency with the logp method
        if weights_ndim_batch:
            new_size = at.concatenate(
                [
                    weights.shape[:weights_ndim_batch],
                    components[0].shape[:ndim_batch],
                ]
            )
            components = cls._resize_components(new_size, *components)

            # Extract support and batch ndims from components and weights
            ndim_batch = components[0].ndim - ndim_supp
            if single_component:
                ndim_batch -= 1
            weights_ndim_batch = max(0, weights.ndim - ndim_batch - 1)

        assert weights_ndim_batch == 0

        # Create a OpFromGraph that encapsulates the random generating process
        # Create dummy input variables with the same type as the ones provided
        weights_ = weights.type()
        components_ = [component.type() for component in components]
        mix_indexes_rng_ = mix_indexes_rng.type()

        mix_axis = -ndim_supp - 1

        # Stack components across mixture axis
        if single_component:
            # If single component, we consider it as being already "stacked"
            stacked_components_ = components_[0]
        else:
            stacked_components_ = at.stack(components_, axis=mix_axis)

        # Broadcast weights to (*batched dimensions, stack dimension), ignoring support dimensions
        weights_broadcast_shape_ = stacked_components_.shape[: ndim_batch + 1]
        weights_broadcasted_ = at.broadcast_to(weights_, weights_broadcast_shape_)

        # Draw mixture indexes and append (stack + ndim_supp) broadcastable dimensions to the right
        mix_indexes_ = at.random.categorical(weights_broadcasted_, rng=mix_indexes_rng_)
        mix_indexes_padded_ = at.shape_padright(mix_indexes_, ndim_supp + 1)

        # Index components and squeeze mixture dimension
        mix_out_ = at.take_along_axis(stacked_components_, mix_indexes_padded_, axis=mix_axis)
        # There is a Aeasara bug in squeeze with negative axis
        # this is equivalent to np.squeeze(mix_out_, axis=mix_axis)
        mix_out_ = at.squeeze(mix_out_, axis=mix_out_.ndim + mix_axis)

        # Output mix_indexes rng update so that it can be updated in place
        mix_indexes_rng_next_ = mix_indexes_.owner.outputs[0]

        mix_op = MarginalMixtureRV(
            inputs=[mix_indexes_rng_, weights_, *components_],
            outputs=[mix_indexes_rng_next_, mix_out_],
        )

        # Create the actual MarginalMixture variable
        mix_indexes_rng_next, mix_out = mix_op(mix_indexes_rng, weights, *components)

        # We need to set_default_updates ourselves, because the choices RV is hidden
        # inside OpFromGraph and PyMC will never find it otherwise
        mix_indexes_rng.default_update = mix_indexes_rng_next

        # Reference nodes to facilitate identification in other classmethods
        mix_out.tag.weights = weights
        mix_out.tag.components = components
        mix_out.tag.choices_rng = mix_indexes_rng

        # Component RVs terms are accounted by the Mixture logprob, so they can be
        # safely ignore by Aeppl (this tag prevents UserWarning)
        for component in components:
            component.tag.ignore_logprob = True

        return mix_out

    @classmethod
    def _reseed_components(cls, rngs, *components):
        *components_rngs, mix_indexes_rng = rngs
        assert len(components) == len(components_rngs)
        new_components = []
        for component, component_rng in zip(components, components_rngs):
            component_node = component.owner
            old_rng, *inputs = component_node.inputs
            new_components.append(
                component_node.op.make_node(component_rng, *inputs).default_output()
            )
        return new_components

    @classmethod
    def _resize_components(cls, size, *components):
        if len(components) == 1:
            # If we have a single component, we need to keep the length of the mixture
            # axis intact, because that's what determines the number of mixture components
            mix_axis = -components[0].owner.op.ndim_supp - 1
            mix_size = components[0].shape[mix_axis]
            size = tuple(size) + (mix_size,)

        return [change_rv_size(component, size) for component in components]

    @classmethod
    def ndim_supp(cls, weights, *components):
        # We already checked that all components have the same support dimensionality
        return components[0].owner.op.ndim_supp

    @classmethod
    def change_size(cls, rv, new_size, expand=False):
        weights = rv.tag.weights
        components = rv.tag.components
        rngs = [component.owner.inputs[0] for component in components] + [rv.tag.choices_rng]

        if expand:
            component = rv.tag.components[0]
            # Old size is equal to `shape[:-ndim_supp]`, with care needed for `ndim_supp == 0`
            size_dims = component.ndim - component.owner.op.ndim_supp
            if len(rv.tag.components) == 1:
                # If we have a single component, new size should ignore the mixture axis
                # dimension, as that is not touched by `_resize_components`
                size_dims -= 1
            old_size = components[0].shape[:size_dims]
            new_size = to_tuple(new_size) + tuple(old_size)

        components = cls._resize_components(new_size, *components)

        return cls.rv_op(weights, *components, rngs=rngs, size=None)

    @classmethod
    def graph_rvs(cls, rv):
        # We return rv, which is itself a pseudo RandomVariable, that contains a
        # mix_indexes_ RV in its inner graph. We want super().dist() to generate
        # (components + 1) rngs for us, and it will do so based on how many elements
        # we return here
        return (*rv.tag.components, rv)


@_get_measurable_outputs.register(MarginalMixtureRV)
def _get_measurable_outputs_MarginalMixtureRV(op, node):
    # This tells Aeppl that the second output is the measurable one
    return [node.outputs[1]]


@_logprob.register(MarginalMixtureRV)
def marginal_mixture_logprob(op, values, rng, weights, *components, **kwargs):
    (value,) = values

    # single component
    if len(components) == 1:
        # Need to broadcast value across mixture axis
        mix_axis = -components[0].owner.op.ndim_supp - 1
        components_logp = logp(components[0], at.expand_dims(value, mix_axis))
    else:
        components_logp = at.stack(
            [logp(component, value) for component in components],
            axis=-1,
        )

    mix_logp = at.logsumexp(at.log(weights) + components_logp, axis=-1)

    # Squeeze stack dimension
    # There is a Aeasara bug in squeeze with negative axis
    # mix_logp = at.squeeze(mix_logp, axis=-1)
    mix_logp = at.squeeze(mix_logp, axis=mix_logp.ndim - 1)

    mix_logp = check_parameters(
        mix_logp,
        0 <= weights,
        weights <= 1,
        at.isclose(at.sum(weights, axis=-1), 1),
        msg="0 <= weights <= 1, sum(weights) == 1",
    )

    return mix_logp


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

    def __init__(self, w, mu, sigma=None, tau=None, sd=None, comp_shape=(), *args, **kwargs):
        if sd is not None:
            sigma = sd
        _, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        self.mu = mu = at.as_tensor_variable(mu)
        self.sigma = self.sd = sigma = at.as_tensor_variable(sigma)

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
    comp_dists: PyMC distribution (e.g. `pm.Multinomial.dist(...)`)
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

        super().__init__(defaults=defaults, *args, **kwargs)

    def logp(self, value):
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
        return check_parameters(
            logsumexp(at.log(w) + comp_logp, axis=mixture_axis, keepdims=False),
            w >= 0,
            w <= 1,
            at.allclose(w.sum(axis=mixture_axis - comp_dists_ndim), 1),
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
        # sample_shape = to_tuple(size)
        # mixture_axis = self.mixture_axis
        #
        # # First we draw values for the mixture component weights
        # (w,) = draw_values([self.w], point=point, size=size)
        #
        # # We now draw random choices from those weights.
        # # However, we have to ensure that the number of choices has the
        # # sample_shape present.
        # w_shape = w.shape
        # batch_shape = self.comp_dists.shape[: mixture_axis + 1]
        # param_shape = np.broadcast(np.empty(w_shape), np.empty(batch_shape)).shape
        # event_shape = self.comp_dists.shape[mixture_axis + 1 :]
        #
        # if np.asarray(self.shape).size != 0:
        #     comp_dists_ndim = len(self.comp_dists.shape)
        #
        #     # If event_shape of both comp_dists and supplied shape matches,
        #     # broadcast only batch_shape
        #     # else broadcast the entire given shape with batch_shape.
        #     if list(self.shape[mixture_axis - comp_dists_ndim + 1 :]) == list(event_shape):
        #         dist_shape = np.broadcast(
        #             np.empty(self.shape[:mixture_axis]), np.empty(param_shape[:mixture_axis])
        #         ).shape
        #     else:
        #         dist_shape = np.broadcast(
        #             np.empty(self.shape), np.empty(param_shape[:mixture_axis])
        #         ).shape
        # else:
        #     dist_shape = param_shape[:mixture_axis]
        #
        # # Try to determine the size that must be used to get the mixture
        # # components (i.e. get random choices using w).
        # # 1. There must be size independent choices based on w.
        # # 2. There must also be independent draws for each non singleton axis
        # # of w.
        # # 3. There must also be independent draws for each dimension added by
        # # self.shape with respect to the w.ndim. These usually correspond to
        # # observed variables with batch shapes
        # wsh = (1,) * (len(dist_shape) - len(w_shape) + 1) + w_shape[:mixture_axis]
        # psh = (1,) * (len(dist_shape) - len(param_shape) + 1) + param_shape[:mixture_axis]
        # w_sample_size = []
        # # Loop through the dist_shape to get the conditions 2 and 3 first
        # for i in range(len(dist_shape)):
        #     if dist_shape[i] != psh[i] and wsh[i] == 1:
        #         # self.shape[i] is a non singleton dimension (usually caused by
        #         # observed data)
        #         sh = dist_shape[i]
        #     else:
        #         sh = wsh[i]
        #     w_sample_size.append(sh)
        #
        # if sample_shape is not None and w_sample_size[: len(sample_shape)] != sample_shape:
        #     w_sample_size = sample_shape + tuple(w_sample_size)
        #
        # choices = random_choice(p=w, size=w_sample_size)
        #
        # # We now draw samples from the mixture components random method
        # comp_samples = self.comp_dists.random(point=point, size=size)
        # if comp_samples.shape[: len(sample_shape)] != sample_shape:
        #     comp_samples = np.broadcast_to(
        #         comp_samples,
        #         shape=sample_shape + comp_samples.shape,
        #     )
        #
        # # At this point the shapes of the arrays involved are:
        # # comp_samples.shape = (sample_shape, batch_shape, mixture_axis, event_shape)
        # # choices.shape = (sample_shape, batch_shape)
        # #
        # # To be able to take the choices along the mixture_axis of the
        # # comp_samples, we have to add in dimensions to the right of the
        # # choices array.
        # # We also need to make sure that the batch_shapes of both the comp_samples
        # # and choices broadcast with each other.
        #
        # choices = np.reshape(choices, choices.shape + (1,) * (1 + len(event_shape)))
        #
        # choices, comp_samples = get_broadcastable_dist_samples([choices, comp_samples], size=size)
        #
        # # We now take the choices of the mixture components along the mixture_axis
        # # but we use the negative index representation to be able to handle the
        # # sample_shape
        # samples = np.take_along_axis(
        #     comp_samples, choices, axis=mixture_axis - len(self.comp_dists.shape)
        # )
        #
        # # The `samples` array still has the `mixture_axis`, so we must remove it:
        # output = samples[(..., 0) + (slice(None),) * len(event_shape)]
        # return output

    def _distr_parameters_for_repr(self):
        return []
