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
import warnings

import aesara
import aesara.tensor as at
import numpy as np

from aeppl.abstract import MeasurableVariable, _get_measurable_outputs
from aeppl.logprob import _logcdf, _logprob
from aeppl.transforms import IntervalTransform
from aesara.compile.builders import OpFromGraph
from aesara.graph.basic import equal_computations
from aesara.tensor import TensorVariable
from aesara.tensor.random.op import RandomVariable

from pymc.aesaraf import change_rv_size
from pymc.distributions import transforms
from pymc.distributions.continuous import Normal, get_tau_sigma
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import SymbolicDistribution, _moment, moment
from pymc.distributions.logprob import logcdf, logp
from pymc.distributions.shape_utils import to_tuple
from pymc.distributions.transforms import _default_transform
from pymc.util import check_dist_not_registered
from pymc.vartypes import continuous_types, discrete_types

__all__ = ["Mixture", "NormalMixture"]


class MarginalMixtureRV(OpFromGraph):
    """A placeholder used to specify a log-likelihood for a mixture sub-graph."""

    default_output = 1


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
    w : tensor_like of float
        w >= 0 and w <= 1
        the mixture weights
    comp_dists : iterable of PyMC distributions or single batched distribution
        Distributions should be created via the `.dist()` API. If single distribution is
        passed, the last size dimension (not shape) determines the number of mixture
        components (e.g. `pm.Poisson.dist(..., size=components)`)
        :math:`f_1, \ldots, f_n`

    Examples
    --------
    .. code-block:: python

        # Mixture of 2 Poisson variables
        with pm.Model() as model:
            w = pm.Dirichlet('w', a=np.array([1, 1]))  # 2 mixture weights

            lam1 = pm.Exponential('lam1', lam=1)
            lam2 = pm.Exponential('lam2', lam=1)

            # As we just need the logp, rather than add a RV to the model, we need to call `.dist()`
            # These two forms are equivalent, but the second benefits from vectorization
            components = [
                pm.Poisson.dist(mu=lam1),
                pm.Poisson.dist(mu=lam2),
            ]
            # `shape=(2,)` indicates 2 mixture components
            components = pm.Poisson.dist(mu=pm.math.stack([lam1, lam2]), shape=(2,))

            like = pm.Mixture('like', w=w, comp_dists=components, observed=data)


    .. code-block:: python

        # Mixture of Normal and StudentT variables
        with pm.Model() as model:
            w = pm.Dirichlet('w', a=np.array([1, 1]))  # 2 mixture weights

            mu = pm.Normal("mu", 0, 1)

            components = [
                pm.Normal.dist(mu=mu, sigma=1),
                pm.StudentT.dist(nu=4, mu=mu, sigma=1),
            ]

            like = pm.Mixture('like', w=w, comp_dists=components, observed=data)


    .. code-block:: python

        # Mixture of (5 x 3) Normal variables
        with pm.Model() as model:
            # w is a stack of 5 independent size 3 weight vectors
            # If shape was `(3,)`, the weights would be shared across the 5 replication dimensions
            w = pm.Dirichlet('w', a=np.ones(3), shape=(5, 3))

            # Each of the 3 mixture components has an independent mean
            mu = pm.Normal('mu', mu=np.arange(3), sigma=1, shape=3)

            # These two forms are equivalent, but the second benefits from vectorization
            components = [
                pm.Normal.dist(mu=mu[0], sigma=1, shape=(5,)),
                pm.Normal.dist(mu=mu[1], sigma=1, shape=(5,)),
                pm.Normal.dist(mu=mu[2], sigma=1, shape=(5,)),
            ]
            components = pm.Normal.dist(mu=mu, sigma=1, shape=(5, 3))

            # The mixture is an array of 5 elements
            # Each element can be thought of as an independent scalar mixture of 3
            # components with different means
            like = pm.Mixture('like', w=w, comp_dists=components, observed=data)


    .. code-block:: python

        # Mixture of 2 Dirichlet variables
        with pm.Model() as model:
            w = pm.Dirichlet('w', a=np.ones(2))  # 2 mixture weights

            # These two forms are equivalent, but the second benefits from vectorization
            components = [
                pm.Dirichlet.dist(a=[1, 10, 100], shape=(3,)),
                pm.Dirichlet.dist(a=[100, 10, 1], shape=(3,)),
            ]
            components = pm.Dirichlet.dist(a=[[1, 10, 100], [100, 10, 1]], shape=(2, 3))

            # The mixture is an array of 3 elements
            # Each element comes from only one of the two core Dirichlet components
            like = pm.Mixture('like', w=w, comp_dists=components, observed=data)
    """

    @classmethod
    def dist(cls, w, comp_dists, **kwargs):
        if not isinstance(comp_dists, (tuple, list)):
            # comp_dists is a single component
            comp_dists = [comp_dists]
        elif len(comp_dists) == 1:
            warnings.warn(
                "Single component will be treated as a mixture across the last size dimension.\n"
                "To disable this warning do not wrap the single component inside a list or tuple",
                UserWarning,
            )

        if len(comp_dists) > 1:
            all_continuous = all(comp_dist.dtype in continuous_types for comp_dist in comp_dists)
            all_discrete = all(comp_dist.dtype in discrete_types for comp_dist in comp_dists)

            if not (
                all(comp_dist.dtype in continuous_types for comp_dist in comp_dists)
                or all(comp_dist.dtype in discrete_types for comp_dist in comp_dists)
            ):
                raise ValueError(
                    "All distributions in comp_dists must be either discrete or continuous.\n"
                    "See the following issue for more information: https://github.com/pymc-devs/pymc/issues/4511."
                )

        # Check that components are not associated with a registered variable in the model
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
            components_ndim_supp.add(dist.owner.op.ndim_supp)

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

        single_component = len(components) == 1
        ndim_supp = components[0].owner.op.ndim_supp

        if size is not None:
            components = cls._resize_components(size, *components)
        elif not single_component:
            # We might need to broadcast components when size is not specified
            shape = tuple(at.broadcast_shape(*components))
            size = shape[: len(shape) - ndim_supp]
            components = cls._resize_components(size, *components)

        # Extract replication ndims from components and weights
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
        # There is a Aesara bug in squeeze with negative axis
        # https://github.com/aesara-devs/aesara/issues/830
        # this is equivalent to np.squeeze(mix_out_, axis=mix_axis)
        mix_out_ = at.squeeze(mix_out_, axis=mix_out_.ndim + mix_axis)

        # Output mix_indexes rng update so that it can be updated in place
        mix_indexes_rng_next_ = mix_indexes_.owner.outputs[0]

        mix_op = MarginalMixtureRV(
            inputs=[mix_indexes_rng_, weights_, *components_],
            outputs=[mix_indexes_rng_next_, mix_out_],
        )

        # Create the actual MarginalMixture variable
        mix_out = mix_op(mix_indexes_rng, weights, *components)

        # We need to set_default_updates ourselves, because the choices RV is hidden
        # inside OpFromGraph and PyMC will never find it otherwise
        mix_indexes_rng.default_update = mix_out.owner.outputs[0]

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
    # There is a Aesara bug in squeeze with negative axis
    # https://github.com/aesara-devs/aesara/issues/830
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


@_logcdf.register(MarginalMixtureRV)
def marginal_mixture_logcdf(op, value, rng, weights, *components, **kwargs):

    # single component
    if len(components) == 1:
        # Need to broadcast value across mixture axis
        mix_axis = -components[0].owner.op.ndim_supp - 1
        components_logcdf = logcdf(components[0], at.expand_dims(value, mix_axis))
    else:
        components_logcdf = at.stack(
            [logcdf(component, value) for component in components],
            axis=-1,
        )

    mix_logcdf = at.logsumexp(at.log(weights) + components_logcdf, axis=-1)

    # Squeeze stack dimension
    # There is a Aesara bug in squeeze with negative axis
    # https://github.com/aesara-devs/aesara/issues/830
    # mix_logp = at.squeeze(mix_logp, axis=-1)
    mix_logcdf = at.squeeze(mix_logcdf, axis=mix_logcdf.ndim - 1)

    mix_logcdf = check_parameters(
        mix_logcdf,
        0 <= weights,
        weights <= 1,
        at.isclose(at.sum(weights, axis=-1), 1),
        msg="0 <= weights <= 1, sum(weights) == 1",
    )

    return mix_logcdf


@_moment.register(MarginalMixtureRV)
def marginal_mixture_moment(op, rv, rng, weights, *components):
    ndim_supp = components[0].owner.op.ndim_supp
    weights = at.shape_padright(weights, ndim_supp)
    mix_axis = -ndim_supp - 1

    if len(components) == 1:
        moment_components = moment(components[0])

    else:
        moment_components = at.stack(
            [moment(component) for component in components],
            axis=mix_axis,
        )

    mix_moment = at.sum(weights * moment_components, axis=mix_axis)
    if components[0].dtype in discrete_types:
        mix_moment = at.round(mix_moment)
    return mix_moment


# List of transforms that can be used by Mixture, either because they do not require
# special handling or because we have custom logic to enable them. If new default
# transforms are implemented, this list and function should be updated
allowed_default_mixture_transforms = (
    transforms.CholeskyCovPacked,
    transforms.CircularTransform,
    transforms.IntervalTransform,
    transforms.LogTransform,
    transforms.LogExpM1,
    transforms.LogOddsTransform,
    transforms.Ordered,
    transforms.Simplex,
    transforms.SumTo1,
)


class MixtureTransformWarning(UserWarning):
    pass


@_default_transform.register(MarginalMixtureRV)
def marginal_mixture_default_transform(op, rv):
    def transform_warning():
        warnings.warn(
            f"No safe default transform found for Mixture distribution {rv}. This can "
            "happen when compoments have different supports or default transforms.\n"
            "If appropriate, you can specify a custom transform for more efficient sampling.",
            MixtureTransformWarning,
            stacklevel=2,
        )

    rng, weights, *components = rv.owner.inputs

    default_transforms = [
        _default_transform(component.owner.op, component) for component in components
    ]

    # If there are more than one type of default transforms, we do not apply any
    if len({type(transform) for transform in default_transforms}) != 1:
        transform_warning()
        return None

    default_transform = default_transforms[0]

    if not isinstance(default_transform, allowed_default_mixture_transforms):
        transform_warning()
        return None

    if isinstance(default_transform, IntervalTransform):
        # If there are more than one component, we need to check the IntervalTransform
        # of the components are actually equivalent (e.g., we don't have an
        # Interval(0, 1), and an Interval(0, 2)).
        if len(default_transforms) > 1:
            value = rv.type()
            backward_expressions = [
                transform.backward(value, *component.owner.inputs)
                for transform, component in zip(default_transforms, components)
            ]
            for expr1, expr2 in zip(backward_expressions[:-1], backward_expressions[1:]):
                if not equal_computations([expr1], [expr2]):
                    transform_warning()
                    return None

        # We need to create a new IntervalTransform that expects the Mixture inputs
        args_fn = default_transform.args_fn

        def mixture_args_fn(rng, weights, *components):
            # We checked that the interval transforms of each component are equivalent,
            # so we can just pass the inputs of the first component
            return args_fn(*components[0].owner.inputs)

        return IntervalTransform(args_fn=mixture_args_fn)

    else:
        return default_transform


class NormalMixture:
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
    w : tensor_like of float
        w >= 0 and w <= 1
        the mixture weights
    mu : tensor_like of float
        the component means
    sigma : tensor_like of float
        the component standard deviations
    tau : tensor_like of float
        the component precisions
    comp_shape : shape of the Normal component
        notice that it should be different than the shape
        of the mixture distribution, with the last axis representing
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
                mu=data.mean(),
                sigma=10,
                shape=n_components,
                transform=pm.transforms.ordered,
                initval=[1, 2, 3],
            )
            σ = pm.HalfNormal("σ", sigma=10, shape=n_components)
            weights = pm.Dirichlet("w", np.ones(n_components))

            y = pm.NormalMixture("y", w=weights, mu=μ, sigma=σ, observed=data)
    """

    def __new__(cls, name, w, mu, sigma=None, tau=None, comp_shape=(), **kwargs):
        _, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        return Mixture(name, w, Normal.dist(mu, sigma=sigma, size=comp_shape), **kwargs)

    @classmethod
    def dist(cls, w, mu, sigma=None, tau=None, comp_shape=(), **kwargs):
        _, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        return Mixture.dist(w, Normal.dist(mu, sigma=sigma, size=comp_shape), **kwargs)
