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
import abc
import warnings

from abc import ABCMeta
from collections.abc import Callable

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.graph.basic import Node, ancestors
from pytensor.graph.replace import clone_replace
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import normalize_size_param

from pymc.distributions.continuous import Normal, get_tau_sigma
from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _support_point,
    support_point,
)
from pymc.distributions.multivariate import MvNormal, MvStudentT
from pymc.distributions.shape_utils import (
    _change_dist_size,
    change_dist_size,
    get_support_shape,
    get_support_shape_1d,
    rv_size_is_none,
)
from pymc.exceptions import NotConstantValueError
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.pytensorf import constant_fold
from pymc.util import check_dist_not_registered

__all__ = [
    "RandomWalk",
    "GaussianRandomWalk",
    "MvGaussianRandomWalk",
    "MvStudentTRandomWalk",
    "AR",
    "GARCH11",
    "EulerMaruyama",
]


class RandomWalkRV(SymbolicRandomVariable):
    """RandomWalk Variable."""

    _print_name = ("RandomWalk", "\\operatorname{RandomWalk}")

    @classmethod
    def rv_op(cls, init_dist, innovation_dist, steps, size=None):
        # We don't allow passing `rng` because we don't fully control the rng of the components!
        steps = pt.as_tensor(steps, dtype=int, ndim=0)

        dist_ndim_supp = init_dist.owner.op.ndim_supp
        init_dist_shape = tuple(init_dist.shape)
        init_dist_batch_shape = init_dist_shape[: len(init_dist_shape) - dist_ndim_supp]
        innovation_dist_shape = tuple(innovation_dist.shape)
        innovation_batch_shape = innovation_dist_shape[
            : len(innovation_dist_shape) - dist_ndim_supp
        ]
        ndim_supp = dist_ndim_supp + 1

        size = normalize_size_param(size)

        # If not explicit, size is determined by the shapes of the input distributions
        if rv_size_is_none(size):
            size = pt.broadcast_shape(
                init_dist_batch_shape, innovation_batch_shape, arrays_are_shapes=True
            )

        # Resize input distributions. We will size them to (T, B, S) in order
        # to safely take random draws. We later swap the steps dimension so
        # that the final distribution will follow (B, T, S)
        # init_dist must have shape (1, B, S)
        init_dist = change_dist_size(init_dist, (1, *size))
        # innovation_dist must have shape (T-1, B, S)
        innovation_dist = change_dist_size(innovation_dist, (steps, *size))

        # We can only infer the logp of a dimshuffled variables, if the dimshuffle is
        # done directly on top of a RandomVariable. Because of this we dimshuffle the
        # distributions and only then concatenate them, instead of the other way around.
        # shape = (B, 1, S)
        init_dist_dimswapped = pt.moveaxis(init_dist, 0, -ndim_supp)
        # shape = (B, T-1, S)
        innovation_dist_dimswapped = pt.moveaxis(innovation_dist, 0, -ndim_supp)
        # shape = (B, T, S)
        grw = pt.concatenate([init_dist_dimswapped, innovation_dist_dimswapped], axis=-ndim_supp)
        grw = pt.cumsum(grw, axis=-ndim_supp)

        innov_supp_dims = [f"d{i}" for i in range(dist_ndim_supp)]
        innov_supp_str = ",".join(innov_supp_dims)
        out_supp_str = ",".join(["t", *innov_supp_dims])
        extended_signature = (
            f"({innov_supp_str}),({innov_supp_str}),(s),[rng]->({out_supp_str}),[rng]"
        )
        return RandomWalkRV(
            [init_dist, innovation_dist, steps],
            # We pass steps_ through just so we can keep a reference to it, even though
            # it's no longer needed at this point
            [grw],
            extended_signature=extended_signature,
        )(init_dist, innovation_dist, steps)


class RandomWalk(Distribution):
    r"""RandomWalk Distribution.

    TODO: Expand docstrings
    """

    rv_type = RandomWalkRV
    rv_op = RandomWalkRV.rv_op

    def __new__(cls, *args, innovation_dist, steps=None, **kwargs):
        steps = cls.get_steps(
            innovation_dist=innovation_dist,
            steps=steps,
            shape=None,  # Shape will be checked in `cls.dist`
            dims=kwargs.get("dims"),
            observed=kwargs.get("observed"),
        )

        return super().__new__(cls, *args, innovation_dist=innovation_dist, steps=steps, **kwargs)

    @classmethod
    def dist(cls, init_dist, innovation_dist, steps=None, **kwargs) -> pt.TensorVariable:
        if not (
            isinstance(init_dist, pt.TensorVariable)
            and init_dist.owner is not None
            and isinstance(init_dist.owner.op, RandomVariable | SymbolicRandomVariable)
        ):
            raise TypeError("init_dist must be a distribution variable")
        check_dist_not_registered(init_dist)

        if not (
            isinstance(innovation_dist, pt.TensorVariable)
            and innovation_dist.owner is not None
            and isinstance(innovation_dist.owner.op, RandomVariable | SymbolicRandomVariable)
        ):
            raise TypeError("innovation_dist must be a distribution variable")
        check_dist_not_registered(innovation_dist)

        if init_dist.owner.op.ndim_supp != innovation_dist.owner.op.ndim_supp:
            raise TypeError(
                "init_dist and innovation_dist must have the same support dimensionality"
            )

        # We need to check this, because we clone the variables when we ignore their logprob next
        if init_dist in ancestors([innovation_dist]) or innovation_dist in ancestors([init_dist]):
            raise ValueError("init_dist and innovation_dist must be completely independent")

        steps = cls.get_steps(
            innovation_dist=innovation_dist,
            steps=steps,
            shape=kwargs.get("shape"),
            dims=None,
            observed=None,
        )
        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        steps = pt.as_tensor_variable(steps, dtype=int)

        return super().dist([init_dist, innovation_dist, steps], **kwargs)

    @classmethod
    def get_steps(cls, innovation_dist, steps, shape, dims, observed):
        # We need to know the ndim_supp of the innovation_dist
        if not (
            isinstance(innovation_dist, pt.TensorVariable)
            and innovation_dist.owner is not None
            and isinstance(innovation_dist.owner.op, RandomVariable | SymbolicRandomVariable)
        ):
            raise TypeError("innovation_dist must be a distribution variable")

        dist_ndim_supp = innovation_dist.owner.op.ndim_supp
        dist_shape = tuple(innovation_dist.shape)
        support_shape = None
        if steps is not None:
            support_shape = (steps,) + (dist_shape[len(dist_shape) - dist_ndim_supp :])
        support_shape = get_support_shape(
            support_shape=support_shape,
            shape=shape,
            dims=dims,
            observed=observed,
            support_shape_offset=1,
            ndim_supp=dist_ndim_supp + 1,
        )
        if support_shape is not None:
            steps = support_shape[-dist_ndim_supp - 1]
        return steps


@_change_dist_size.register(RandomWalkRV)
def change_random_walk_size(op, dist, new_size, expand):
    init_dist, innovation_dist, steps = dist.owner.inputs
    if expand:
        old_shape = tuple(dist.shape)
        old_size = old_shape[: len(old_shape) - op.ndim_supp]
        new_size = tuple(new_size) + tuple(old_size)
    return RandomWalk.rv_op(init_dist, innovation_dist, steps, size=new_size)


@_support_point.register(RandomWalkRV)
def random_walk_support_point(op, rv, init_dist, innovation_dist, steps):
    # shape = (1, B, S)
    init_support_point = support_point(init_dist)
    # shape = (T-1, B, S)
    innovation_support_point = support_point(innovation_dist)
    # shape = (T, B, S)
    grw_support_point = pt.concatenate([init_support_point, innovation_support_point], axis=0)
    grw_support_point = pt.cumsum(grw_support_point, axis=0)
    # shape = (B, T, S)
    grw_support_point = pt.moveaxis(grw_support_point, 0, -op.ndim_supp)
    return grw_support_point


@_logprob.register(RandomWalkRV)
def random_walk_logp(op, values, *inputs, **kwargs):
    # Although we can derive the logprob of random walks, it does not collapse
    # what we consider the core dimension of steps. We do it manually here.
    (value,) = values
    # Recreate RV and obtain inner graph
    rv_node = op.make_node(*inputs)
    rv = clone_replace(op.inner_outputs, replace=dict(zip(op.inner_inputs, rv_node.inputs)))[
        op.default_output
    ]
    # Obtain logp of the inner graph and collapse steps dimension
    return logp(rv, value).sum(axis=-1)


class PredefinedRandomWalk(ABCMeta):
    """Base class for predefined RandomWalk distributions."""

    def __new__(cls, name, *args, **kwargs):
        init_dist, innovation_dist, kwargs = cls.get_dists(*args, **kwargs)
        return RandomWalk(name, init_dist=init_dist, innovation_dist=innovation_dist, **kwargs)

    @classmethod
    def dist(cls, *args, **kwargs) -> pt.TensorVariable:
        init_dist, innovation_dist, kwargs = cls.get_dists(*args, **kwargs)
        return RandomWalk.dist(init_dist=init_dist, innovation_dist=innovation_dist, **kwargs)

    @classmethod
    @abc.abstractmethod
    def get_dists(cls, *args, **kwargs):
        pass


class GaussianRandomWalk(PredefinedRandomWalk):
    r"""Random Walk with Normal innovations.

    Parameters
    ----------
    mu : tensor_like of float, default 0
        innovation drift
    sigma : tensor_like of float, default 1
        sigma > 0, innovation standard deviation.
    init_dist : unnamed_distribution
        Unnamed univariate distribution of the initial value. Unnamed refers to distributions
        created with the ``.dist()`` API.

        .. warning:: init_dist will be cloned, rendering them independent of the ones passed as input.

    steps : int, optional
        Number of steps in Gaussian Random Walk (steps > 0). Only needed if shape is not
        provided.
    """

    @classmethod
    def get_dists(cls, mu=0.0, sigma=1.0, *, init_dist=None, **kwargs):
        if "init" in kwargs:
            warnings.warn(
                "init parameter is now called init_dist. Using init will raise an error in a future release.",
                FutureWarning,
            )
            init_dist = kwargs.pop("init")

        if init_dist is None:
            warnings.warn(
                "Initial distribution not specified, defaulting to `Normal.dist(0, 100)`."
                "You can specify an init_dist manually to suppress this warning.",
                UserWarning,
            )
            init_dist = Normal.dist(0, 100)

        mu = pt.as_tensor_variable(mu)
        sigma = pt.as_tensor_variable(sigma)
        innovation_dist = Normal.dist(mu=mu, sigma=sigma)

        return init_dist, innovation_dist, kwargs


class MvGaussianRandomWalk(PredefinedRandomWalk):
    r"""Random Walk with Multivariate Normal innovations.

    Parameters
    ----------
    mu : tensor_like of float
        innovation drift
    cov : tensor_like of float
        pos def matrix, innovation covariance matrix
    tau : tensor_like of float
        pos def matrix, inverse covariance matrix
    chol : tensor_like of float
        Cholesky decomposition of covariance matrix
    lower : bool, default=True
        Whether the cholesky fatcor is given as a lower triangular matrix.
    init_dist : unnamed_distribution
        Unnamed multivariate distribution of the initial value.

         .. warning:: init_dist will be cloned, rendering them independent of the ones passed as input.

    steps : int, optional
        Number of steps in Random Walk (steps > 0). Only needed if shape is not
        provided.

    Notes
    -----
    Only one of cov, tau or chol is required.

    """

    @classmethod
    def get_dists(cls, mu, *, cov=None, tau=None, chol=None, lower=True, init_dist=None, **kwargs):
        if "init" in kwargs:
            warnings.warn(
                "init parameter is now called init_dist. Using init will raise an error "
                "in a future release.",
                FutureWarning,
            )
            init_dist = kwargs.pop("init")

        if init_dist is None:
            warnings.warn(
                "Initial distribution not specified, defaulting to `MvNormal.dist(0, I*100)`."
                "You can specify an init_dist manually to suppress this warning.",
                UserWarning,
            )
            init_dist = MvNormal.dist(pt.zeros_like(mu.shape[-1]), pt.eye(mu.shape[-1]) * 100)

        innovation_dist = MvNormal.dist(mu=mu, cov=cov, tau=tau, chol=chol, lower=lower)
        return init_dist, innovation_dist, kwargs


class MvStudentTRandomWalk(PredefinedRandomWalk):
    r"""Multivariate Random Walk with StudentT innovations.

    Parameters
    ----------
    nu : int
        degrees of freedom
    mu : tensor_like of float
        innovation drift
    scale : tensor_like of float
        pos def matrix, innovation covariance matrix
    tau : tensor_like of float
        pos def matrix, inverse covariance matrix
    chol : tensor_like of float
        Cholesky decomposition of covariance matrix
    lower : bool, default=True
        Whether the cholesky fatcor is given as a lower triangular matrix.
    init_dist : unnamed_distribution
        Unnamed multivariate distribution of the initial value.

         .. warning:: init_dist will be cloned, rendering them independent of the ones passed as input.

    steps : int, optional
        Number of steps in Random Walk (steps > 0). Only needed if shape is not
        provided.

    Notes
    -----
    Only one of cov, tau or chol is required.

    """

    @classmethod
    def get_dists(
        cls, *, nu, mu, scale=None, tau=None, chol=None, lower=True, init_dist=None, **kwargs
    ):
        if "init" in kwargs:
            warnings.warn(
                "init parameter is now called init_dist. Using init will raise an error "
                "in a future release.",
                FutureWarning,
            )
            init_dist = kwargs.pop("init")

        if init_dist is None:
            warnings.warn(
                "Initial distribution not specified, defaulting to `MvNormal.dist(0, I*100)`."
                "You can specify an init_dist manually to suppress this warning.",
                UserWarning,
            )
            init_dist = MvNormal.dist(pt.zeros_like(mu.shape[-1]), pt.eye(mu.shape[-1]) * 100)

        innovation_dist = MvStudentT.dist(
            nu=nu, mu=mu, scale=scale, tau=tau, chol=chol, lower=lower, cov=kwargs.pop("cov", None)
        )
        return init_dist, innovation_dist, kwargs


class AutoRegressiveRV(SymbolicRandomVariable):
    """A placeholder used to specify a log-likelihood for an AR sub-graph."""

    extended_signature = "(o),(),(o),(s),[rng]->[rng],(t)"
    ar_order: int
    constant_term: bool
    _print_name = ("AR", "\\operatorname{AR}")

    def __init__(self, *args, ar_order, constant_term, **kwargs):
        self.ar_order = ar_order
        self.constant_term = constant_term
        super().__init__(*args, **kwargs)

    @classmethod
    def rv_op(cls, rhos, sigma, init_dist, steps, ar_order, constant_term, size=None):
        # We don't allow passing `rng` because we don't fully control the rng of the components!
        noise_rng = pytensor.shared(np.random.default_rng())
        size = normalize_size_param(size)

        # Init dist should have shape (*size, ar_order)
        if rv_size_is_none(size):
            # In this case the size of the init_dist depends on the parameters shape
            # The last dimension of rho and init_dist does not matter
            batch_size = pt.broadcast_shape(
                tuple(sigma.shape),
                tuple(rhos.shape)[:-1],
                tuple(pt.atleast_1d(init_dist).shape)[:-1],
                arrays_are_shapes=True,
            )
        else:
            batch_size = size

        if init_dist.owner.op.ndim_supp == 0:
            init_dist_size = (*batch_size, ar_order)
        else:
            # In this case the support dimension must cover for ar_order
            init_dist_size = batch_size
        init_dist = change_dist_size(init_dist, init_dist_size)

        rhos_bcast_shape = init_dist.shape
        if constant_term:
            # In this case init shape is one unit smaller than rhos in the last dimension
            rhos_bcast_shape = (*rhos_bcast_shape[:-1], rhos_bcast_shape[-1] + 1)
        rhos_bcast = pt.broadcast_to(rhos, rhos_bcast_shape)

        def step(*args):
            *prev_xs, reversed_rhos, sigma, rng = args
            if constant_term:
                mu = reversed_rhos[-1] + pt.sum(prev_xs * reversed_rhos[:-1], axis=0)
            else:
                mu = pt.sum(prev_xs * reversed_rhos, axis=0)
            next_rng, new_x = Normal.dist(mu=mu, sigma=sigma, rng=rng).owner.outputs
            return new_x, {rng: next_rng}

        # We transpose inputs as scan iterates over first dimension
        innov, innov_updates = pytensor.scan(
            fn=step,
            outputs_info=[{"initial": init_dist.T, "taps": range(-ar_order, 0)}],
            non_sequences=[rhos_bcast.T[::-1], sigma.T, noise_rng],
            n_steps=steps,
            strict=True,
        )
        (noise_next_rng,) = tuple(innov_updates.values())
        ar = pt.concatenate([init_dist, innov.T], axis=-1)

        return AutoRegressiveRV(
            inputs=[rhos, sigma, init_dist, steps, noise_rng],
            outputs=[noise_next_rng, ar],
            ar_order=ar_order,
            constant_term=constant_term,
        )(rhos, sigma, init_dist, steps, noise_rng)

    def update(self, node: Node):
        """Return the update mapping for the noise RV."""
        return {node.inputs[-1]: node.outputs[0]}


class AR(Distribution):
    r"""Autoregressive process with p lags.

    .. math::

       x_t = \rho_0 + \rho_1 x_{t-1} + \ldots + \rho_p x_{t-p} + \epsilon_t,
       \epsilon_t \sim N(0,\sigma^2)

    The innovation can be parameterized either in terms of precision or standard
    deviation. The link between the two parametrizations is given by

    .. math::

       \tau = \dfrac{1}{\sigma^2}

    Parameters
    ----------
    rho : tensor_like of float
        Tensor of autoregressive coefficients. The n-th entry in the last dimension is
        the coefficient for the n-th lag.
    sigma : tensor_like of float, default 1
        Standard deviation of innovation (sigma > 0). Only required if
        tau is not specified.
    tau : tensor_like of float, optional
        Precision of innovation (tau > 0).
    constant : bool, default False
        Whether the first element of rho should be used as a constant term in the AR
        process.
    init_dist : unnamed_distribution, optional
        Scalar or vector distribution for initial values. Distributions should have shape (*shape[:-1], ar_order).
        If not, it will be automatically resized. Defaults to pm.Normal.dist(0, 100, shape=...).

        .. warning:: init_dist will be cloned, rendering it independent of the one passed as input.

    ar_order : int, optional
        Order of the AR process. Inferred from length of the last dimension of rho, if
        possible. ar_order = rho.shape[-1] if constant else rho.shape[-1] - 1
    steps : int, optional
        Number of steps in AR process (steps > 0). Only needed if shape is not provided.

    Notes
    -----
    The init distribution will be cloned, rendering it distinct from the one passed as
    input.

    Examples
    --------
    .. code-block:: python

        # Create an AR of order 3, with a constant term
        with pm.Model() as AR3:
            # The first coefficient will be the constant term
            coefs = pm.Normal("coefs", 0, size=4)
            # We need one init variable for each lag, hence size=3
            init = pm.Normal.dist(5, size=3)
            ar3 = pm.AR("ar3", coefs, sigma=1.0, init_dist=init, constant=True, steps=500)

    """

    rv_type = AutoRegressiveRV
    rv_op = AutoRegressiveRV.rv_op

    def __new__(cls, name, rho, *args, steps=None, constant=False, ar_order=None, **kwargs):
        rhos = pt.atleast_1d(pt.as_tensor_variable(rho))
        ar_order = cls._get_ar_order(rhos=rhos, constant=constant, ar_order=ar_order)
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,  # Shape will be checked in `cls.dist`
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            support_shape_offset=ar_order,
        )
        return super().__new__(
            cls, name, rhos, *args, steps=steps, constant=constant, ar_order=ar_order, **kwargs
        )

    @classmethod
    def dist(
        cls,
        rho,
        sigma=None,
        tau=None,
        *,
        init_dist=None,
        steps=None,
        constant=False,
        ar_order=None,
        **kwargs,
    ):
        _, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        sigma = pt.as_tensor_variable(sigma)
        rhos = pt.atleast_1d(pt.as_tensor_variable(rho))

        if "init" in kwargs:
            warnings.warn(
                "init parameter is now called init_dist. Using init will raise an error in a future release.",
                FutureWarning,
            )
            init_dist = kwargs.pop("init")

        ar_order = cls._get_ar_order(rhos=rhos, constant=constant, ar_order=ar_order)
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=ar_order
        )
        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        steps = pt.as_tensor_variable(steps, dtype=int, ndim=0)

        if init_dist is not None:
            if not isinstance(init_dist, TensorVariable) or not isinstance(
                init_dist.owner.op, RandomVariable | SymbolicRandomVariable
            ):
                raise ValueError(
                    f"Init dist must be a distribution created via the `.dist()` API, "
                    f"got {type(init_dist)}"
                )
            check_dist_not_registered(init_dist)
            if init_dist.owner.op.ndim_supp > 1:
                raise ValueError(
                    "Init distribution must have a scalar or vector support dimension, ",
                    f"got ndim_supp={init_dist.owner.op.ndim_supp}.",
                )
        else:
            warnings.warn(
                "Initial distribution not specified, defaulting to "
                "`Normal.dist(0, 100, shape=...)`. You can specify an init_dist "
                "manually to suppress this warning.",
                UserWarning,
            )
            init_dist = Normal.dist(0, 100, shape=(*sigma.shape, ar_order))

        return super().dist([rhos, sigma, init_dist, steps, ar_order, constant], **kwargs)

    @classmethod
    def _get_ar_order(cls, rhos: TensorVariable, ar_order: int | None, constant: bool) -> int:
        """Compute ar_order given inputs.

        If ar_order is not specified we do constant folding on the shape of rhos
        to retrieve it. For example, this will detect that
        Normal(size=(5, 3)).shape[-1] == 3, which is not known by PyTensor before.

        Raises
        ------
        ValueError
            If inferred ar_order cannot be inferred from rhos or if it is less than 1
        """
        if ar_order is None:
            try:
                (folded_shape,) = constant_fold((rhos.shape[-1],))
            except NotConstantValueError:
                raise ValueError(
                    "Could not infer ar_order from last dimension of rho. Pass it "
                    "explicitly or make sure rho have a static shape"
                )
            ar_order = int(folded_shape) - int(constant)
            if ar_order < 1:
                raise ValueError(
                    "Inferred ar_order is smaller than 1. Increase the last dimension "
                    "of rho or remove constant_term"
                )

        return ar_order


@_change_dist_size.register(AutoRegressiveRV)
def change_ar_size(op, dist, new_size, expand=False):
    if expand:
        old_size = dist.shape[:-1]
        new_size = tuple(new_size) + tuple(old_size)

    return AR.rv_op(
        *dist.owner.inputs[:-1],
        ar_order=op.ar_order,
        constant_term=op.constant_term,
        size=new_size,
    )


@_logprob.register(AutoRegressiveRV)
def ar_logp(op, values, rhos, sigma, init_dist, steps, noise_rng, **kwargs):
    (value,) = values

    ar_order = op.ar_order
    constant_term = op.constant_term

    # Convolve rhos with values
    if constant_term:
        expectation = pt.add(
            rhos[..., 0, None],
            *(
                rhos[..., i + 1, None] * value[..., ar_order - (i + 1) : -(i + 1)]
                for i in range(ar_order)
            ),
        )
    else:
        expectation = pt.add(
            *(
                rhos[..., i, None] * value[..., ar_order - (i + 1) : -(i + 1)]
                for i in range(ar_order)
            )
        )
    # Compute and collapse logp across time dimension
    innov_logp = pt.sum(
        logp(Normal.dist(0, sigma[..., None]), value[..., ar_order:] - expectation), axis=-1
    )
    init_logp = logp(init_dist, value[..., :ar_order])
    if init_dist.owner.op.ndim_supp == 0:
        init_logp = pt.sum(init_logp, axis=-1)
    return init_logp + innov_logp


@_support_point.register(AutoRegressiveRV)
def ar_support_point(op, rv, rhos, sigma, init_dist, steps, noise_rng):
    # Use last entry of init_dist support_point as the moment for the whole AR
    return pt.full_like(rv, support_point(init_dist)[..., -1, None])


class GARCH11RV(SymbolicRandomVariable):
    """A placeholder used to specify a GARCH11 graph."""

    extended_signature = "(),(),(),(),(),(s),[rng]->[rng],(t)"
    _print_name = ("GARCH11", "\\operatorname{GARCH11}")

    @classmethod
    def rv_op(cls, omega, alpha_1, beta_1, initial_vol, init_dist, steps, size=None):
        # We don't allow passing `rng` because we don't fully control the rng of the components!
        steps = pt.as_tensor(steps, ndim=0)
        omega = pt.as_tensor(omega)
        alpha_1 = pt.as_tensor(alpha_1)
        beta_1 = pt.as_tensor(beta_1)
        initial_vol = pt.as_tensor(initial_vol)
        noise_rng = pytensor.shared(np.random.default_rng())
        size = normalize_size_param(size)

        if rv_size_is_none(size):
            # In this case the size of the init_dist depends on the parameters shape
            batch_size = pt.broadcast_shape(omega, alpha_1, beta_1, initial_vol)
        else:
            batch_size = size

        init_dist = change_dist_size(init_dist, batch_size)

        # Create OpFromGraph representing random draws from GARCH11 process

        def step(prev_y, prev_sigma, omega, alpha_1, beta_1, rng):
            new_sigma = pt.sqrt(
                omega + alpha_1 * pt.square(prev_y) + beta_1 * pt.square(prev_sigma)
            )
            next_rng, new_y = Normal.dist(mu=0, sigma=new_sigma, rng=rng).owner.outputs
            return (new_y, new_sigma), {rng: next_rng}

        (y_t, _), innov_updates = pytensor.scan(
            fn=step,
            outputs_info=[
                init_dist,
                pt.broadcast_to(initial_vol.astype("floatX"), init_dist.shape),
            ],
            non_sequences=[omega, alpha_1, beta_1, noise_rng],
            n_steps=steps,
            strict=True,
        )
        (noise_next_rng,) = tuple(innov_updates.values())

        garch11 = pt.concatenate([init_dist[None, ...], y_t], axis=0).dimshuffle(
            (*range(1, y_t.ndim), 0)
        )

        return GARCH11RV(
            inputs=[omega, alpha_1, beta_1, initial_vol, init_dist, steps, noise_rng],
            outputs=[noise_next_rng, garch11],
        )(omega, alpha_1, beta_1, initial_vol, init_dist, steps, noise_rng)

    def update(self, node: Node):
        """Return the update mapping for the noise RV."""
        return {node.inputs[-1]: node.outputs[0]}


class GARCH11(Distribution):
    r"""
    GARCH(1,1) with Normal innovations. The model is specified by.

    .. math::
        y_t \sim N(0, \sigma_t^2)

    .. math::
        \sigma_t^2 = \omega + \alpha_1 * y_{t-1}^2 + \beta_1 * \sigma_{t-1}^2

    where \sigma_t^2 (the error variance) follows a ARMA(1, 1) model.

    Parameters
    ----------
    omega : tensor_like of float
        omega > 0, mean variance
    alpha_1 : tensor_like of float
        alpha_1 >= 0, autoregressive term coefficient
    beta_1 : tensor_like of float
        beta_1 >= 0, alpha_1 + beta_1 < 1, moving average term coefficient
    initial_vol : tensor_like of float
        initial_vol >= 0, initial volatility, sigma_0
    """

    rv_type = GARCH11RV
    rv_op = GARCH11RV.rv_op

    def __new__(cls, *args, steps=None, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,  # Shape will be checked in `cls.dist`
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            support_shape_offset=1,
        )
        return super().__new__(cls, *args, steps=steps, **kwargs)

    @classmethod
    def dist(cls, omega, alpha_1, beta_1, initial_vol, *, steps=None, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=1
        )
        if steps is None:
            raise ValueError("Must specify steps or shape parameter")

        init_dist = Normal.dist(0, initial_vol)
        return super().dist([omega, alpha_1, beta_1, initial_vol, init_dist, steps], **kwargs)


@_change_dist_size.register(GARCH11RV)
def change_garch11_size(op, dist, new_size, expand=False):
    if expand:
        old_size = dist.shape[:-1]
        new_size = tuple(new_size) + tuple(old_size)

    return GARCH11.rv_op(
        *dist.owner.inputs[:-1],
        size=new_size,
    )


@_logprob.register(GARCH11RV)
def garch11_logp(
    op, values, omega, alpha_1, beta_1, initial_vol, init_dist, steps, noise_rng, **kwargs
):
    (value,) = values
    # Move the time axis to the first dimension
    value_dimswapped = value.dimshuffle((value.ndim - 1, *range(0, value.ndim - 1)))
    initial_vol = initial_vol * pt.ones_like(value_dimswapped[0])

    def volatility_update(x, vol, w, a, b):
        return pt.sqrt(w + a * pt.square(x) + b * pt.square(vol))

    vol, _ = pytensor.scan(
        fn=volatility_update,
        sequences=[value_dimswapped[:-1]],
        outputs_info=[initial_vol],
        non_sequences=[omega, alpha_1, beta_1],
        strict=True,
    )
    sigma_t = pt.concatenate([[initial_vol], vol])
    # Compute and collapse logp across time dimension
    innov_logp = pt.sum(logp(Normal.dist(0, sigma_t), value_dimswapped), axis=0)
    return innov_logp


@_support_point.register(GARCH11RV)
def garch11_support_point(op, rv, omega, alpha_1, beta_1, initial_vol, init_dist, steps, noise_rng):
    # GARCH(1,1) mean is zero
    return pt.zeros_like(rv)


class EulerMaruyamaRV(SymbolicRandomVariable):
    """A placeholder used to specify a log-likelihood for a EulerMaruyama sub-graph."""

    dt: float
    sde_fn: Callable
    _print_name = ("EulerMaruyama", "\\operatorname{EulerMaruyama}")

    def __init__(self, *args, dt: float, sde_fn: Callable, **kwargs):
        self.dt = dt
        self.sde_fn = sde_fn
        super().__init__(*args, **kwargs)

    @classmethod
    def rv_op(cls, init_dist, steps, sde_pars, dt, sde_fn, size=None):
        # We don't allow passing `rng` because we don't fully control the rng of the components!
        noise_rng = pytensor.shared(np.random.default_rng())

        # Init dist should have shape (*size,)
        if size is not None:
            batch_size = size
        else:
            batch_size = pt.broadcast_shape(*sde_pars, init_dist)
        init_dist = change_dist_size(init_dist, batch_size)

        # Create OpFromGraph representing random draws from SDE process
        def step(*prev_args):
            prev_y, *prev_sde_pars, rng = prev_args
            f, g = sde_fn(prev_y, *prev_sde_pars)
            mu = prev_y + dt * f
            sigma = pt.sqrt(dt) * g
            next_rng, next_y = Normal.dist(mu=mu, sigma=sigma, rng=rng).owner.outputs
            return next_y, {rng: next_rng}

        y_t, innov_updates = pytensor.scan(
            fn=step,
            outputs_info=[init_dist],
            non_sequences=[*sde_pars, noise_rng],
            n_steps=steps,
            strict=True,
        )
        (noise_next_rng,) = tuple(innov_updates.values())

        sde_out = pt.concatenate([init_dist[None, ...], y_t], axis=0).dimshuffle(
            (*range(1, y_t.ndim), 0)
        )

        return EulerMaruyamaRV(
            inputs=[init_dist, steps, *sde_pars, noise_rng],
            outputs=[noise_next_rng, sde_out],
            dt=dt,
            sde_fn=sde_fn,
            extended_signature=f"(),(s),{','.join('()' for _ in sde_pars)},[rng]->[rng],(t)",
        )(init_dist, steps, *sde_pars, noise_rng)

    def update(self, node: Node):
        """Return the update mapping for the noise RV."""
        return {node.inputs[-1]: node.outputs[0]}


class EulerMaruyama(Distribution):
    r"""
    Stochastic differential equation discretized with the Euler-Maruyama method.

    Parameters
    ----------
    dt : float
        time step of discretization
    sde_fn : callable
        function returning the drift and diffusion coefficients of SDE
    sde_pars : tuple
        parameters of the SDE, passed as ``*args`` to ``sde_fn``
    init_dist : unnamed_distribution, optional
        Scalar distribution for initial values. Distributions should have shape (*shape[:-1]).
        If not, it will be automatically resized. Defaults to pm.Normal.dist(0, 100, shape=...).

        .. warning:: init_dist will be cloned, rendering it independent of the one passed as input.
    """

    rv_type = EulerMaruyamaRV
    rv_op = EulerMaruyamaRV.rv_op

    def __new__(cls, name, dt, sde_fn, *args, steps=None, **kwargs):
        dt = pt.as_tensor_variable(dt)
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,  # Shape will be checked in `cls.dist`
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            support_shape_offset=1,
        )
        return super().__new__(cls, name, dt, sde_fn, *args, steps=steps, **kwargs)

    @classmethod
    def dist(cls, dt, sde_fn, sde_pars, *, init_dist=None, steps=None, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=1
        )
        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        steps = pt.as_tensor_variable(steps, dtype=int, ndim=0)

        dt = pt.as_tensor_variable(dt)
        sde_pars = [pt.as_tensor_variable(x) for x in sde_pars]

        if init_dist is not None:
            if not isinstance(init_dist, TensorVariable) or not isinstance(
                init_dist.owner.op, RandomVariable | SymbolicRandomVariable
            ):
                raise ValueError(
                    f"Init dist must be a distribution created via the `.dist()` API, "
                    f"got {type(init_dist)}"
                )
            check_dist_not_registered(init_dist)
            if init_dist.owner.op.ndim_supp > 0:
                raise ValueError(
                    "Init distribution must have a scalar support dimension, ",
                    f"got ndim_supp={init_dist.owner.op.ndim_supp}.",
                )
        else:
            warnings.warn(
                "Initial distribution not specified, defaulting to "
                "`Normal.dist(0, 100, shape=...)`. You can specify an init_dist "
                "manually to suppress this warning.",
                UserWarning,
            )
            init_dist = Normal.dist(0, 100, shape=sde_pars[0].shape)

        return super().dist([init_dist, steps, sde_pars, dt, sde_fn], **kwargs)


@_change_dist_size.register(EulerMaruyamaRV)
def change_eulermaruyama_size(op, dist, new_size, expand=False):
    if expand:
        old_size = dist.shape[:-1]
        new_size = tuple(new_size) + tuple(old_size)

    init_dist, steps, *sde_pars, _ = dist.owner.inputs
    return EulerMaruyama.rv_op(
        init_dist,
        steps,
        sde_pars,
        dt=op.dt,
        sde_fn=op.sde_fn,
        size=new_size,
    )


@_logprob.register(EulerMaruyamaRV)
def eulermaruyama_logp(op, values, init_dist, steps, *sde_pars_noise_arg, **kwargs):
    (x,) = values
    # noise arg is unused, but is needed to make the logp signature match the rv_op signature
    *sde_pars, _ = sde_pars_noise_arg
    # sde_fn is user provided and likely not broadcastable to additional time dimension,
    # since the input x is now [..., t], we need to broadcast each input to [..., None]
    # below as best effort attempt to make it work
    sde_pars_broadcast = [x[..., None] for x in sde_pars]
    xtm1 = x[..., :-1]
    xt = x[..., 1:]
    f, g = op.sde_fn(xtm1, *sde_pars_broadcast)
    mu = xtm1 + op.dt * f
    sigma = pt.sqrt(op.dt) * g
    # Compute and collapse logp across time dimension
    sde_logp = pt.sum(logp(Normal.dist(mu, sigma), xt), axis=-1)
    init_logp = logp(init_dist, x[..., 0])
    return init_logp + sde_logp
