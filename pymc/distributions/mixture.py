#   Copyright 2024 The PyMC Developers
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
import itertools
import warnings

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.graph.basic import Apply, equal_computations
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import normalize_size_param

from pymc.distributions import transforms
from pymc.distributions.continuous import Gamma, LogNormal, Normal, get_tau_sigma
from pymc.distributions.discrete import Binomial, NegativeBinomial, Poisson
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import (
    DiracDelta,
    Distribution,
    SymbolicRandomVariable,
    _support_point,
    support_point,
)
from pymc.distributions.shape_utils import _change_dist_size, change_dist_size, rv_size_is_none
from pymc.distributions.transforms import _default_transform
from pymc.distributions.truncated import Truncated
from pymc.logprob.abstract import _logcdf, _logcdf_helper, _logprob
from pymc.logprob.basic import logp
from pymc.logprob.transforms import IntervalTransform
from pymc.util import check_dist_not_registered
from pymc.vartypes import continuous_types, discrete_types

__all__ = [
    "HurdleGamma",
    "HurdleLogNormal",
    "HurdleNegativeBinomial",
    "HurdlePoisson",
    "Mixture",
    "NormalMixture",
    "ZeroInflatedBinomial",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedPoisson",
]


class MarginalMixtureRV(SymbolicRandomVariable):
    """A placeholder used to specify a log-likelihood for a mixture sub-graph."""

    _print_name = ("MarginalMixture", "\\operatorname{MarginalMixture}")

    @classmethod
    def rv_op(cls, weights, *components, size=None):
        # We don't allow passing `rng` because we don't fully control the rng of the components!
        mix_indexes_rng = pytensor.shared(np.random.default_rng())

        single_component = len(components) == 1
        ndim_supp = components[0].owner.op.ndim_supp

        size = normalize_size_param(size)
        if not rv_size_is_none(size):
            components = cls._resize_components(size, *components)
        elif not single_component:
            # We might need to broadcast components when size is not specified
            shape = tuple(pt.broadcast_shape(*components))
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
            new_size = pt.concatenate(
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

        mix_axis = -ndim_supp - 1

        # Stack components across mixture axis
        if single_component:
            # If single component, we consider it as being already "stacked"
            stacked_components = components[0]
        else:
            stacked_components = pt.stack(components, axis=mix_axis)

        # Broadcast weights to (*batched dimensions, stack dimension), ignoring support dimensions
        weights_broadcast_shape = stacked_components.shape[: ndim_batch + 1]
        weights_broadcasted = pt.broadcast_to(weights, weights_broadcast_shape)

        # Draw mixture indexes and append (stack + ndim_supp) broadcastable dimensions to the right
        mix_indexes_rng_next, mix_indexes = pt.random.categorical(
            weights_broadcasted, rng=mix_indexes_rng
        ).owner.outputs
        mix_indexes_padded = pt.shape_padright(mix_indexes, ndim_supp + 1)

        # Index components and squeeze mixture dimension
        mix_out = pt.take_along_axis(stacked_components, mix_indexes_padded, axis=mix_axis)
        mix_out = pt.squeeze(mix_out, axis=mix_axis)

        s = ",".join(f"s{i}" for i in range(components[0].owner.op.ndim_supp))
        if len(components) == 1:
            comp_s = ",".join((*s, "w"))
            extended_signature = f"[rng],(w),({comp_s})->[rng],({s})"
        else:
            comps_s = ",".join(f"({s})" for _ in components)
            extended_signature = f"[rng],(w),{comps_s}->[rng],({s})"

        return MarginalMixtureRV(
            inputs=[mix_indexes_rng, weights, *components],
            outputs=[mix_indexes_rng_next, mix_out],
            extended_signature=extended_signature,
        )(mix_indexes_rng, weights, *components)

    @classmethod
    def _resize_components(cls, size, *components):
        if len(components) == 1:
            # If we have a single component, we need to keep the length of the mixture
            # axis intact, because that's what determines the number of mixture components
            mix_axis = -components[0].owner.op.ndim_supp - 1
            mix_size = components[0].shape[mix_axis]
            size = (*size, mix_size)

        return [change_dist_size(component, size) for component in components]

    def update(self, node: Apply):
        # Update for the internal mix_indexes RV
        return {node.inputs[0]: node.outputs[0]}


class Mixture(Distribution):
    R"""
    Mixture log-likelihood.

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
    comp_dists : iterable of unnamed distributions or single batched distribution
        Distributions should be created via the `.dist()` API. If a single distribution
        is passed, the last size dimension (not shape) determines the number of mixture
        components (e.g. `pm.Poisson.dist(..., size=components)`)
        :math:`f_1, \ldots, f_n`

        .. warning:: comp_dists will be cloned, rendering them independent of the ones passed as input.


    Examples
    --------
    .. code-block:: python

        # Mixture of 2 Poisson variables
        with pm.Model() as model:
            w = pm.Dirichlet("w", a=np.array([1, 1]))  # 2 mixture weights

            lam1 = pm.Exponential("lam1", lam=1)
            lam2 = pm.Exponential("lam2", lam=1)

            # As we just need the logp, rather than add a RV to the model, we need to call `.dist()`
            # These two forms are equivalent, but the second benefits from vectorization
            components = [
                pm.Poisson.dist(mu=lam1),
                pm.Poisson.dist(mu=lam2),
            ]
            # `shape=(2,)` indicates 2 mixture components
            components = pm.Poisson.dist(mu=pm.math.stack([lam1, lam2]), shape=(2,))

            like = pm.Mixture("like", w=w, comp_dists=components, observed=data)


    .. code-block:: python

        # Mixture of Normal and StudentT variables
        with pm.Model() as model:
            w = pm.Dirichlet("w", a=np.array([1, 1]))  # 2 mixture weights

            mu = pm.Normal("mu", 0, 1)

            components = [
                pm.Normal.dist(mu=mu, sigma=1),
                pm.StudentT.dist(nu=4, mu=mu, sigma=1),
            ]

            like = pm.Mixture("like", w=w, comp_dists=components, observed=data)


    .. code-block:: python

        # Mixture of (5 x 3) Normal variables
        with pm.Model() as model:
            # w is a stack of 5 independent size 3 weight vectors
            # If shape was `(3,)`, the weights would be shared across the 5 replication dimensions
            w = pm.Dirichlet("w", a=np.ones(3), shape=(5, 3))

            # Each of the 3 mixture components has an independent mean
            mu = pm.Normal("mu", mu=np.arange(3), sigma=1, shape=3)

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
            like = pm.Mixture("like", w=w, comp_dists=components, observed=data)


    .. code-block:: python

        # Mixture of 2 Dirichlet variables
        with pm.Model() as model:
            w = pm.Dirichlet("w", a=np.ones(2))  # 2 mixture weights

            # These two forms are equivalent, but the second benefits from vectorization
            components = [
                pm.Dirichlet.dist(a=[1, 10, 100], shape=(3,)),
                pm.Dirichlet.dist(a=[100, 10, 1], shape=(3,)),
            ]
            components = pm.Dirichlet.dist(a=[[1, 10, 100], [100, 10, 1]], shape=(2, 3))

            # The mixture is an array of 3 elements
            # Each element comes from only one of the two core Dirichlet components
            like = pm.Mixture("like", w=w, comp_dists=components, observed=data)
    """

    rv_type = MarginalMixtureRV
    rv_op = MarginalMixtureRV.rv_op

    @classmethod
    def dist(cls, w, comp_dists, **kwargs):
        if not isinstance(comp_dists, tuple | list):
            # comp_dists is a single component
            comp_dists = [comp_dists]
        elif len(comp_dists) == 1:
            warnings.warn(
                "Single component will be treated as a mixture across the last size dimension.\n"
                "To disable this warning do not wrap the single component inside a list or tuple",
                UserWarning,
            )

        if len(comp_dists) > 1:
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
                dist.owner.op, RandomVariable | SymbolicRandomVariable
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

        w = pt.as_tensor_variable(w)
        return super().dist([w, *comp_dists], **kwargs)


@_change_dist_size.register(MarginalMixtureRV)
def change_marginal_mixture_size(op, dist, new_size, expand=False):
    rng, weights, *components = dist.owner.inputs

    if expand:
        component = components[0]
        # Old size is equal to `shape[:-ndim_supp]`, with care needed for `ndim_supp == 0`
        size_dims = component.ndim - component.owner.op.ndim_supp
        if len(components) == 1:
            # If we have a single component, new size should ignore the mixture axis
            # dimension, as that is not touched by `_resize_components`
            size_dims -= 1
        old_size = components[0].shape[:size_dims]
        new_size = tuple(new_size) + tuple(old_size)

    return Mixture.rv_op(weights, *components, size=new_size)


@_logprob.register(MarginalMixtureRV)
def marginal_mixture_logprob(op, values, rng, weights, *components, **kwargs):
    (value,) = values

    # single component
    if len(components) == 1:
        # Need to broadcast value across mixture axis
        mix_axis = -components[0].owner.op.ndim_supp - 1
        components_logp = logp(components[0], pt.expand_dims(value, mix_axis))
    else:
        components_logp = pt.stack(
            [logp(component, value) for component in components],
            axis=-1,
        )

    mix_logp = pt.logsumexp(pt.log(weights) + components_logp, axis=-1)

    mix_logp = check_parameters(
        mix_logp,
        0 <= weights,
        weights <= 1,
        pt.isclose(pt.sum(weights, axis=-1), 1),
        msg="0 <= weights <= 1, sum(weights) == 1",
    )

    return mix_logp


@_logcdf.register(MarginalMixtureRV)
def marginal_mixture_logcdf(op, value, rng, weights, *components, **kwargs):
    # single component
    if len(components) == 1:
        # Need to broadcast value across mixture axis
        mix_axis = -components[0].owner.op.ndim_supp - 1
        components_logcdf = _logcdf_helper(components[0], pt.expand_dims(value, mix_axis))
    else:
        components_logcdf = pt.stack(
            [_logcdf_helper(component, value) for component in components],
            axis=-1,
        )

    mix_logcdf = pt.logsumexp(pt.log(weights) + components_logcdf, axis=-1)

    mix_logcdf = check_parameters(
        mix_logcdf,
        0 <= weights,
        weights <= 1,
        pt.isclose(pt.sum(weights, axis=-1), 1),
        msg="0 <= weights <= 1, sum(weights) == 1",
    )

    return mix_logcdf


@_support_point.register(MarginalMixtureRV)
def marginal_mixture_support_point(op, rv, rng, weights, *components):
    ndim_supp = components[0].owner.op.ndim_supp
    weights = pt.shape_padright(weights, ndim_supp)
    mix_axis = -ndim_supp - 1

    if len(components) == 1:
        support_point_components = support_point(components[0])

    else:
        support_point_components = pt.stack(
            [support_point(component) for component in components],
            axis=mix_axis,
        )

    mix_support_point = pt.sum(weights * support_point_components, axis=mix_axis)
    if components[0].dtype in discrete_types:
        mix_support_point = pt.round(mix_support_point)
    return mix_support_point


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
    transforms.SimplexTransform,
    transforms.SumTo1,
)


class MixtureTransformWarning(UserWarning):
    pass


@_default_transform.register(MarginalMixtureRV)
def marginal_mixture_default_transform(op, rv):
    def transform_warning():
        warnings.warn(
            f"No safe default transform found for Mixture distribution {rv}. This can "
            "happen when components have different supports or default transforms.\n"
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

    if default_transform is None:
        return None

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
            for expr1, expr2 in itertools.pairwise(backward_expressions):
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
    Normal mixture log-likelihood.

    .. math::

        f(x \mid w, \mu, \sigma^2) = \sum_{i = 1}^n w_i N(x \mid \mu_i, \sigma^2_i)

    ========  =======================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\sum_{i = 1}^n w_i \mu_i`
    Variance  :math:`\sum_{i = 1}^n w_i (\sigma^2_i + \mu_i^2) - \left(\sum_{i = 1}^n w_i \mu_i\right)^2`
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
                transform=pm.distributions.transforms.ordered,
                initval=[1, 2, 3],
            )
            σ = pm.HalfNormal("σ", sigma=10, shape=n_components)
            weights = pm.Dirichlet("w", np.ones(n_components))

            y = pm.NormalMixture("y", w=weights, mu=μ, sigma=σ, observed=data)
    """

    def __new__(cls, name, w, mu, sigma=None, tau=None, **kwargs):
        _, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        return Mixture(name, w, Normal.dist(mu, sigma=sigma), **kwargs)

    @classmethod
    def dist(cls, w, mu, sigma=None, tau=None, **kwargs):
        _, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        return Mixture.dist(w, Normal.dist(mu, sigma=sigma), **kwargs)


def _zero_inflated_mixture(*, name, nonzero_p, nonzero_dist, **kwargs):
    """Create a zero-inflated mixture (helper function).

    If name is `None`, this function returns an unregistered variable.
    """
    nonzero_p = pt.as_tensor_variable(nonzero_p)
    weights = pt.stack([1 - nonzero_p, nonzero_p], axis=-1)
    comp_dists = [
        DiracDelta.dist(0),
        nonzero_dist,
    ]
    if name is not None:
        return Mixture(name, weights, comp_dists, **kwargs)
    else:
        return Mixture.dist(weights, comp_dists, **kwargs)


class ZeroInflatedPoisson:
    R"""
    Zero-inflated Poisson log-likelihood.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math::

        f(x \mid \psi, \mu) = \left\{ \begin{array}{l}
            (1-\psi) + \psi e^{-\mu}, \text{if } x = 0 \\
            \psi \frac{e^{-\mu}\mu^x}{x!}, \text{if } x=1,2,3,\ldots
            \end{array} \right.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.arange(0, 22)
        psis = [0.7, 0.4]
        mus = [8, 4]
        for psi, mu in zip(psis, mus):
            pmf = st.poisson.pmf(x, mu)
            pmf[0] = (1 - psi) + pmf[0]
            pmf[1:] =  psi * pmf[1:]
            pmf /= pmf.sum()
            plt.plot(x, pmf, '-o', label='$\\psi$ = {}, $\\mu$ = {}'.format(psi, mu))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi\mu`
    Variance  :math:`\mu + \frac{1-\psi}{\psi}\mu^2`
    ========  ==========================

    Parameters
    ----------
    psi : tensor_like of float
        Expected proportion of Poisson draws (0 < psi < 1)
    mu : tensor_like of float
        Expected number of occurrences during the given interval
        (mu >= 0).
    """

    def __new__(cls, name, psi, mu, **kwargs):
        return _zero_inflated_mixture(
            name=name, nonzero_p=psi, nonzero_dist=Poisson.dist(mu=mu), **kwargs
        )

    @classmethod
    def dist(cls, psi, mu, **kwargs):
        return _zero_inflated_mixture(
            name=None, nonzero_p=psi, nonzero_dist=Poisson.dist(mu=mu), **kwargs
        )


class ZeroInflatedBinomial:
    R"""
    Zero-inflated Binomial log-likelihood.

    The pmf of this distribution is

    .. math::

        f(x \mid \psi, n, p) = \left\{ \begin{array}{l}
            (1-\psi) + \psi (1-p)^{n}, \text{if } x = 0 \\
            \psi {n \choose x} p^x (1-p)^{n-x}, \text{if } x=1,2,3,\ldots,n
            \end{array} \right.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.arange(0, 25)
        ns = [10, 20]
        ps = [0.5, 0.7]
        psis = [0.7, 0.4]
        for n, p, psi in zip(ns, ps, psis):
            pmf = st.binom.pmf(x, n, p)
            pmf[0] = (1 - psi) + pmf[0]
            pmf[1:] =  psi * pmf[1:]
            pmf /= pmf.sum()
            plt.plot(x, pmf, '-o', label='n = {}, p = {}, $\\psi$ = {}'.format(n, p, psi))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi n p`
    Variance  :math:`(1-\psi) n p [1 - p(1 - \psi n)].`
    ========  ==========================

    Parameters
    ----------
    psi : tensor_like of float
        Expected proportion of Binomial draws (0 < psi < 1)
    n : tensor_like of int
        Number of Bernoulli trials (n >= 0).
    p : tensor_like of float
        Probability of success in each trial (0 < p < 1).
    """

    def __new__(cls, name, psi, n, p, **kwargs):
        return _zero_inflated_mixture(
            name=name, nonzero_p=psi, nonzero_dist=Binomial.dist(n=n, p=p), **kwargs
        )

    @classmethod
    def dist(cls, psi, n, p, **kwargs):
        return _zero_inflated_mixture(
            name=None, nonzero_p=psi, nonzero_dist=Binomial.dist(n=n, p=p), **kwargs
        )


class ZeroInflatedNegativeBinomial:
    R"""
    Zero-Inflated Negative binomial log-likelihood.

    The Zero-inflated version of the Negative Binomial (NB).
    The NB distribution describes a Poisson random variable
    whose rate parameter is gamma distributed.
    The pmf of this distribution is.

    .. math::

       f(x \mid \psi, \mu, \alpha) = \left\{
         \begin{array}{l}
           (1-\psi) + \psi \left (
             \frac{\alpha}{\alpha+\mu}
           \right) ^\alpha, \text{if } x = 0 \\
           \psi \frac{\Gamma(x+\alpha)}{x! \Gamma(\alpha)} \left (
             \frac{\alpha}{\mu+\alpha}
           \right)^\alpha \left(
             \frac{\mu}{\mu+\alpha}
           \right)^x, \text{if } x=1,2,3,\ldots
         \end{array}
       \right.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        from scipy import special
        import arviz as az
        plt.style.use('arviz-darkgrid')
        def ZeroInfNegBinom(a, m, psi, x):
            pmf = special.binom(x + a - 1, x) * (a / (m + a))**a * (m / (m + a))**x
            pmf[0] = (1 - psi) + pmf[0]
            pmf[1:] =  psi * pmf[1:]
            pmf /= pmf.sum()
            return pmf
        x = np.arange(0, 25)
        alphas = [2, 4]
        mus = [2, 8]
        psis = [0.7, 0.7]
        for a, m, psi in zip(alphas, mus, psis):
            pmf = ZeroInfNegBinom(a, m, psi, x)
            plt.plot(x, pmf, '-o', label=r'$\alpha$ = {}, $\mu$ = {}, $\psi$ = {}'.format(a, m, psi))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi\mu`
    Var       .. math::
                  \psi \left(\frac{{\mu^2}}{{\alpha}}\right) +\
                  \psi \mu + \psi \mu^2 - \psi^2 \mu^2
    ========  ==========================

    The zero inflated negative binomial distribution can be parametrized
    either in terms of mu or p, and either in terms of alpha or n.
    The link between the parametrizations is given by

    .. math::

        \mu &= \frac{n(1-p)}{p} \\
        \alpha &= n

    Parameters
    ----------
    psi : tensor_like of float
        Expected proportion of NegativeBinomial draws (0 < psi < 1)
    mu : tensor_like of float
        Poisson distribution parameter (mu > 0).
    alpha : tensor_like of float
        Gamma distribution parameter (alpha > 0).
    p : tensor_like of float
        Alternative probability of success in each trial (0 < p < 1).
    n : tensor_like of float
        Alternative number of target success trials (n > 0)
    """

    def __new__(cls, name, psi, mu=None, alpha=None, p=None, n=None, **kwargs):
        return _zero_inflated_mixture(
            name=name,
            nonzero_p=psi,
            nonzero_dist=NegativeBinomial.dist(mu=mu, alpha=alpha, p=p, n=n),
            **kwargs,
        )

    @classmethod
    def dist(cls, psi, mu=None, alpha=None, p=None, n=None, **kwargs):
        return _zero_inflated_mixture(
            name=None,
            nonzero_p=psi,
            nonzero_dist=NegativeBinomial.dist(mu=mu, alpha=alpha, p=p, n=n),
            **kwargs,
        )


def _hurdle_mixture(*, name, nonzero_p, nonzero_dist, dtype, max_n_steps=10_000, **kwargs):
    """Create a hurdle mixtures (helper function).

    If name is `None`, this function returns an unregistered variable

    In hurdle models, the zeros come from a completely different process than the rest of the data.
    In other words, the zeros are not inflated, they come from a different process.
    """
    if dtype == "float":
        zero = 0.0
        lower = np.finfo(pytensor.config.floatX).eps
    elif dtype == "int":
        zero = 0
        lower = 1
    else:
        raise ValueError("dtype must be 'float' or 'int'")

    nonzero_p = pt.as_tensor_variable(nonzero_p)
    weights = pt.stack([1 - nonzero_p, nonzero_p], axis=-1)
    comp_dists = [
        DiracDelta.dist(zero),
        Truncated.dist(nonzero_dist, lower=lower, max_n_steps=max_n_steps),
    ]

    if name is not None:
        return Mixture(name, weights, comp_dists, **kwargs)
    else:
        return Mixture.dist(weights, comp_dists, **kwargs)


class HurdlePoisson:
    R"""
    Hurdle Poisson log-likelihood.

    The Poisson distribution is often used to model the number of events occurring
    in a fixed period of time or space when the times or locations
    at which events occur are independent.

    The difference with ZeroInflatedPoisson is that the zeros are not inflated,
    they come from a completely independent process.

    The pmf of this distribution is

    .. math::

        f(x \mid \psi, \mu) =
            \left\{
                \begin{array}{l}
                (1 - \psi)  \ \text{if } x = 0 \\
                \psi
                \frac{\text{PoissonPDF}(x \mid \mu))}
                {1 - \text{PoissonCDF}(0 \mid \mu)} \ \text{if } x=1,2,3,\ldots
                \end{array}
            \right.


    Parameters
    ----------
    psi : tensor_like of float
        Expected proportion of Poisson draws (0 < psi < 1)
    mu : tensor_like of float
        Expected number of occurrences (mu >= 0).
    """

    def __new__(cls, name, psi, mu, **kwargs):
        return _hurdle_mixture(
            name=name, nonzero_p=psi, nonzero_dist=Poisson.dist(mu=mu), dtype="int", **kwargs
        )

    @classmethod
    def dist(cls, psi, mu, **kwargs):
        return _hurdle_mixture(
            name=None, nonzero_p=psi, nonzero_dist=Poisson.dist(mu=mu), dtype="int", **kwargs
        )


class HurdleNegativeBinomial:
    R"""
    Hurdle Negative Binomial log-likelihood.

    The negative binomial distribution describes a Poisson random variable
    whose rate parameter is gamma distributed.

    The difference with ZeroInflatedNegativeBinomial is that the zeros are not inflated,
    they come from a completely independent process.

    The pmf of this distribution is

    .. math::

        f(x \mid \psi, \mu, \alpha) =
            \left\{
                \begin{array}{l}
                (1 - \psi)  \ \text{if } x = 0 \\
                \psi
                \frac{\text{NegativeBinomialPDF}(x \mid \mu, \alpha))}
                {1 - \text{NegativeBinomialCDF}(0 \mid \mu, \alpha)} \ \text{if } x=1,2,3,\ldots
                \end{array}
            \right.

    Parameters
    ----------
    psi : tensor_like of float
        Expected proportion of Negative Binomial draws (0 < psi < 1)
    alpha : tensor_like of float
        Gamma distribution shape parameter (alpha > 0).
    mu : tensor_like of float
        Gamma distribution mean (mu > 0).
    p : tensor_like of float
        Alternative probability of success in each trial (0 < p < 1).
    n : tensor_like of float
        Alternative number of target success trials (n > 0)
    """

    def __new__(cls, name, psi, mu=None, alpha=None, p=None, n=None, **kwargs):
        return _hurdle_mixture(
            name=name,
            nonzero_p=psi,
            nonzero_dist=NegativeBinomial.dist(mu=mu, alpha=alpha, p=p, n=n),
            dtype="int",
            **kwargs,
        )

    @classmethod
    def dist(cls, psi, mu=None, alpha=None, p=None, n=None, **kwargs):
        return _hurdle_mixture(
            name=None,
            nonzero_p=psi,
            nonzero_dist=NegativeBinomial.dist(mu=mu, alpha=alpha, p=p, n=n),
            dtype="int",
            **kwargs,
        )


class HurdleGamma:
    R"""
    Hurdle Gamma log-likelihood.

    .. math::

        f(x \mid \psi, \alpha, \beta) =
            \left\{
                \begin{array}{l}
                (1 - \psi)  \ \text{if } x = 0 \\
                \psi
                \frac{\text{GammaPDF}(x \mid \alpha, \beta))}
                {1 - \text{GammaCDF}(\epsilon \mid \alpha, \beta)} \ \text{if } x=1,2,3,\ldots
                \end{array}
            \right.

    where :math:`\epsilon` is the machine precision.

    Parameters
    ----------
    psi : tensor_like of float
        Expected proportion of Gamma draws (0 < psi < 1)
    alpha : tensor_like of float, optional
        Shape parameter (alpha > 0).
    beta : tensor_like of float, optional
        Rate parameter (beta > 0).
    mu : tensor_like of float, optional
        Alternative shape parameter (mu > 0).
    sigma : tensor_like of float, optional
        Alternative scale parameter (sigma > 0).
    """

    def __new__(cls, name, psi, alpha=None, beta=None, mu=None, sigma=None, **kwargs):
        return _hurdle_mixture(
            name=name,
            nonzero_p=psi,
            nonzero_dist=Gamma.dist(alpha=alpha, beta=beta, mu=mu, sigma=sigma),
            dtype="float",
            **kwargs,
        )

    @classmethod
    def dist(cls, psi, alpha=None, beta=None, mu=None, sigma=None, **kwargs):
        return _hurdle_mixture(
            name=None,
            nonzero_p=psi,
            nonzero_dist=Gamma.dist(alpha=alpha, beta=beta, mu=mu, sigma=sigma),
            dtype="float",
            **kwargs,
        )


class HurdleLogNormal:
    R"""
    Hurdle LogNormal log-likelihood.

    .. math::

        f(x \mid \psi, \mu, \sigma) =
            \left\{
                \begin{array}{l}
                (1 - \psi)  \ \text{if } x = 0 \\
                \psi
                \frac{\text{LogNormalPDF}(x \mid \mu, \sigma))}
                {1 - \text{LogNormalCDF}(\epsilon \mid \mu, \sigma)} \ \text{if } x=1,2,3,\ldots
                \end{array}
            \right.

    where :math:`\epsilon` is the machine precision.

    Parameters
    ----------
    psi : tensor_like of float
        Expected proportion of LogNormal draws (0 < psi < 1)
    mu : tensor_like of float, default 0
        Location parameter.
    sigma : tensor_like of float, optional
        Standard deviation. (sigma > 0). (only required if tau is not specified).
        Defaults to 1.
    tau : tensor_like of float, optional
        Scale parameter (tau > 0). (only required if sigma is not specified).
        Defaults to 1.
    """

    def __new__(cls, name, psi, mu=0, sigma=None, tau=None, **kwargs):
        return _hurdle_mixture(
            name=name,
            nonzero_p=psi,
            nonzero_dist=LogNormal.dist(mu=mu, sigma=sigma, tau=tau),
            dtype="float",
            **kwargs,
        )

    @classmethod
    def dist(cls, psi, mu=0, sigma=None, tau=None, **kwargs):
        return _hurdle_mixture(
            name=None,
            nonzero_p=psi,
            nonzero_dist=LogNormal.dist(mu=mu, sigma=sigma, tau=tau),
            dtype="float",
            **kwargs,
        )
