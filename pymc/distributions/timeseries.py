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

from typing import Optional

import aesara
import aesara.tensor as at
import numpy as np

from aeppl.abstract import _get_measurable_outputs
from aeppl.logprob import _logprob
from aesara.graph.basic import Node, clone_replace
from aesara.tensor import TensorVariable
from aesara.tensor.random.op import RandomVariable

from pymc.aesaraf import constant_fold, convert_observed_data, floatX, intX
from pymc.distributions import distribution, multivariate
from pymc.distributions.continuous import Flat, Normal, get_tau_sigma
from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _moment,
    moment,
)
from pymc.distributions.logprob import ignore_logprob, logp
from pymc.distributions.shape_utils import (
    _change_dist_size,
    change_dist_size,
    get_support_shape_1d,
    to_tuple,
)
from pymc.exceptions import NotConstantValueError
from pymc.model import modelcontext
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
    """RandomWalk Variable"""

    default_output = 0
    _print_name = ("RandomWalk", "\\operatorname{RandomWalk}")


class RandomWalk(Distribution):
    r"""RandomWalk Distribution

    TODO: Expand docstrings
    """

    rv_type = RandomWalkRV

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
    def dist(cls, init_dist, innovation_dist, steps=None, **kwargs) -> at.TensorVariable:
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=kwargs.get("shape"),
            support_shape_offset=1,
        )
        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        steps = at.as_tensor_variable(intX(steps))

        if not (
            isinstance(init_dist, at.TensorVariable)
            and init_dist.owner is not None
            and isinstance(init_dist.owner.op, (RandomVariable, SymbolicRandomVariable))
            # TODO: Lift univariate constraint on init_dist
            and init_dist.owner.op.ndim_supp == 0
        ):
            raise TypeError("init_dist must be a univariate distribution variable")
        check_dist_not_registered(init_dist)

        if not (
            isinstance(innovation_dist, at.TensorVariable)
            and init_dist.owner is not None
            and isinstance(init_dist.owner.op, (RandomVariable, SymbolicRandomVariable))
            # TODO: Lift univariate constraint on inovvation_dist
            and init_dist.owner.op.ndim_supp == 0
        ):
            raise TypeError("init_dist must be a univariate distribution variable")
        check_dist_not_registered(init_dist)

        return super().dist([init_dist, innovation_dist, steps], **kwargs)

    @classmethod
    def rv_op(cls, init_dist, innovation_dist, steps, size=None):
        if not steps.ndim == 0 or not steps.dtype.startswith("int"):
            raise ValueError("steps must be an integer scalar (ndim=0).")

        # If not explicit, size is determined by the shapes of the input distributions
        if size is None:
            size = at.broadcast_shape(init_dist, innovation_dist[..., 0])
        innovation_size = tuple(size) + (steps,)

        # Resize input distributions
        init_dist = change_dist_size(init_dist, size)
        innovation_dist = change_dist_size(innovation_dist, innovation_size)

        # Create SymbolicRV
        init_dist_, innovation_dist_, steps_ = (
            init_dist.type(),
            innovation_dist.type(),
            steps.type(),
        )
        grw_ = at.concatenate([init_dist_[..., None], innovation_dist_], axis=-1)
        grw_ = at.cumsum(grw_, axis=-1)
        return RandomWalkRV(
            [init_dist_, innovation_dist_, steps_],
            # We pass steps_ through just so we can keep a reference to it, even though
            # it's no longer needed at this point
            [grw_, steps_],
            ndim_supp=1,
        )(init_dist, innovation_dist, steps)


@_get_measurable_outputs.register(RandomWalkRV)
def _get_measurable_outputs_random_walk(op, node):
    # Ignore steps output
    return [node.default_output()]


@_change_dist_size.register(RandomWalkRV)
def change_random_walk_size(op, dist, new_size, expand):
    init_dist, innovation_dist, steps = dist.owner.inputs
    if expand:
        old_size = init_dist.shape
        new_size = tuple(new_size) + tuple(old_size)
    return RandomWalk.rv_op(init_dist, innovation_dist, steps, size=new_size)


@_moment.register(RandomWalkRV)
def random_walk_moment(op, rv, init_dist, innovation_dist, steps):
    grw_moment = at.zeros_like(rv)
    grw_moment = at.set_subtensor(grw_moment[..., 0], moment(init_dist))
    grw_moment = at.set_subtensor(grw_moment[..., 1:], moment(innovation_dist))
    return at.cumsum(grw_moment, axis=-1)


@_logprob.register(RandomWalkRV)
def random_walk_logp(op, values, *inputs, **kwargs):
    # Although Aeppl can derive the logprob of random walks, it does not collapse
    # what PyMC considers the core dimension of steps. We do it manually here.
    (value,) = values
    # Recreate RV and obtain inner graph
    rv_node = op.make_node(*inputs)
    rv = clone_replace(
        op.inner_outputs, replace={u: v for u, v in zip(op.inner_inputs, rv_node.inputs)}
    )[op.default_output]
    # Obtain logp via Aeppl of inner graph and collapse steps dimension
    return logp(rv, value).sum(axis=-1)


class GaussianRandomWalk:
    r"""Random Walk with Normal innovations.

    Parameters
    ----------
    mu : tensor_like of float, default 0
        innovation drift
    sigma : tensor_like of float, default 1
        sigma > 0, innovation standard deviation.
    init_dist : Distribution
        Unnamed univariate distribution of the initial value. Unnamed refers to distributions
         created with the ``.dist()`` API.

        .. warning:: init will be cloned, rendering them independent of the ones passed as input.

    steps : int, optional
        Number of steps in Gaussian Random Walk (steps > 0). Only needed if shape is not
        provided.
    """

    def __new__(cls, name, mu=0.0, sigma=1.0, *, init_dist=None, steps=None, **kwargs):
        init_dist, innovation_dist, kwargs = cls.get_dists(
            mu=mu, sigma=sigma, init_dist=init_dist, **kwargs
        )
        return RandomWalk(
            name, init_dist=init_dist, innovation_dist=innovation_dist, steps=steps, **kwargs
        )

    @classmethod
    def dist(cls, mu=0.0, sigma=1.0, *, init_dist=None, steps=None, **kwargs) -> at.TensorVariable:
        init_dist, innovation_dist, kwargs = cls.get_dists(
            mu=mu, sigma=sigma, init_dist=init_dist, **kwargs
        )
        return RandomWalk.dist(
            init_dist=init_dist, innovation_dist=innovation_dist, steps=steps, **kwargs
        )

    @classmethod
    def get_dists(cls, *, mu, sigma, init_dist, **kwargs):
        if "init" in kwargs:
            warnings.warn(
                "init parameter is now called init_dist. Using init will raise an error in a future release.",
                FutureWarning,
            )
            init_dist = kwargs.pop("init")

        # If no scalar distribution is passed then initialize with a Normal of same mu and sigma
        if init_dist is None:
            warnings.warn(
                "Initial distribution not specified, defaulting to `Normal.dist(0, 100)`."
                "You can specify an init_dist manually to suppress this warning.",
                UserWarning,
            )
            init_dist = Normal.dist(0, 100)

        # Add one dimension to the right, so that mu and sigma broadcast safely along
        # the steps dimension
        mu = at.as_tensor_variable(mu)
        sigma = at.as_tensor_variable(sigma)
        innovation_dist = Normal.dist(mu=mu[..., None], sigma=sigma[..., None])

        return init_dist, innovation_dist, kwargs


class AutoRegressiveRV(SymbolicRandomVariable):
    """A placeholder used to specify a log-likelihood for an AR sub-graph."""

    default_output = 1
    ar_order: int
    constant_term: bool
    _print_name = ("AR", "\\operatorname{AR}")

    def __init__(self, *args, ar_order, constant_term, **kwargs):
        self.ar_order = ar_order
        self.constant_term = constant_term
        super().__init__(*args, **kwargs)

    def update(self, node: Node):
        """Return the update mapping for the noise RV."""
        # Since noise is a shared variable it shows up as the last node input
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
    init_dist : unnamed distribution, optional
        Scalar or vector distribution for initial values. Unnamed refers to distributions
         created with the ``.dist()`` API. Distributions should have shape (*shape[:-1], ar_order).
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

    def __new__(cls, name, rho, *args, steps=None, constant=False, ar_order=None, **kwargs):
        rhos = at.atleast_1d(at.as_tensor_variable(floatX(rho)))
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
        sigma = at.as_tensor_variable(floatX(sigma))
        rhos = at.atleast_1d(at.as_tensor_variable(floatX(rho)))

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
        steps = at.as_tensor_variable(intX(steps), ndim=0)

        if init_dist is not None:
            if not isinstance(init_dist, TensorVariable) or not isinstance(
                init_dist.owner.op, (RandomVariable, SymbolicRandomVariable)
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

        # Tell Aeppl to ignore init_dist, as it will be accounted for in the logp term
        init_dist = ignore_logprob(init_dist)

        return super().dist([rhos, sigma, init_dist, steps, ar_order, constant], **kwargs)

    @classmethod
    def _get_ar_order(cls, rhos: TensorVariable, ar_order: Optional[int], constant: bool) -> int:
        """Compute ar_order given inputs

        If ar_order is not specified we do constant folding on the shape of rhos
        to retrieve it. For example, this will detect that
        Normal(size=(5, 3)).shape[-1] == 3, which is not known by Aesara before.

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
                    "explictily or make sure rho have a static shape"
                )
            ar_order = int(folded_shape) - int(constant)
            if ar_order < 1:
                raise ValueError(
                    "Inferred ar_order is smaller than 1. Increase the last dimension "
                    "of rho or remove constant_term"
                )

        return ar_order

    @classmethod
    def ndim_supp(cls, *args):
        return 1

    @classmethod
    def rv_op(cls, rhos, sigma, init_dist, steps, ar_order, constant_term, size=None):
        # Init dist should have shape (*size, ar_order)
        if size is not None:
            batch_size = size
        else:
            # In this case the size of the init_dist depends on the parameters shape
            # The last dimension of rho and init_dist does not matter
            batch_size = at.broadcast_shape(sigma, rhos[..., 0], at.atleast_1d(init_dist)[..., 0])
        if init_dist.owner.op.ndim_supp == 0:
            init_dist_size = (*batch_size, ar_order)
        else:
            # In this case the support dimension must cover for ar_order
            init_dist_size = batch_size
        init_dist = change_dist_size(init_dist, init_dist_size)

        # Create OpFromGraph representing random draws form AR process
        # Variables with underscore suffix are dummy inputs into the OpFromGraph
        init_ = init_dist.type()
        rhos_ = rhos.type()
        sigma_ = sigma.type()
        steps_ = steps.type()

        rhos_bcast_shape_ = init_.shape
        if constant_term:
            # In this case init shape is one unit smaller than rhos in the last dimension
            rhos_bcast_shape_ = (*rhos_bcast_shape_[:-1], rhos_bcast_shape_[-1] + 1)
        rhos_bcast_ = at.broadcast_to(rhos_, rhos_bcast_shape_)

        noise_rng = aesara.shared(np.random.default_rng())

        def step(*args):
            *prev_xs, reversed_rhos, sigma, rng = args
            if constant_term:
                mu = reversed_rhos[-1] + at.sum(prev_xs * reversed_rhos[:-1], axis=0)
            else:
                mu = at.sum(prev_xs * reversed_rhos, axis=0)
            next_rng, new_x = Normal.dist(mu=mu, sigma=sigma, rng=rng).owner.outputs
            return new_x, {rng: next_rng}

        # We transpose inputs as scan iterates over first dimension
        innov_, innov_updates_ = aesara.scan(
            fn=step,
            outputs_info=[{"initial": init_.T, "taps": range(-ar_order, 0)}],
            non_sequences=[rhos_bcast_.T[::-1], sigma_.T, noise_rng],
            n_steps=steps_,
            strict=True,
        )
        (noise_next_rng,) = tuple(innov_updates_.values())
        ar_ = at.concatenate([init_, innov_.T], axis=-1)

        ar_op = AutoRegressiveRV(
            inputs=[rhos_, sigma_, init_, steps_],
            outputs=[noise_next_rng, ar_],
            ar_order=ar_order,
            constant_term=constant_term,
            ndim_supp=1,
        )

        ar = ar_op(rhos, sigma, init_dist, steps)
        return ar


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
        expectation = at.add(
            rhos[..., 0, None],
            *(
                rhos[..., i + 1, None] * value[..., ar_order - (i + 1) : -(i + 1)]
                for i in range(ar_order)
            ),
        )
    else:
        expectation = at.add(
            *(
                rhos[..., i, None] * value[..., ar_order - (i + 1) : -(i + 1)]
                for i in range(ar_order)
            )
        )
    # Compute and collapse logp across time dimension
    innov_logp = at.sum(
        logp(Normal.dist(0, sigma[..., None]), value[..., ar_order:] - expectation), axis=-1
    )
    init_logp = logp(init_dist, value[..., :ar_order])
    if init_dist.owner.op.ndim_supp == 0:
        init_logp = at.sum(init_logp, axis=-1)
    return init_logp + innov_logp


@_moment.register(AutoRegressiveRV)
def ar_moment(op, rv, rhos, sigma, init_dist, steps, noise_rng):
    # Use last entry of init_dist moment as the moment for the whole AR
    return at.full_like(rv, moment(init_dist)[..., -1, None])


class GARCH11RV(SymbolicRandomVariable):
    """A placeholder used to specify a GARCH11 graph."""

    default_output = 1
    _print_name = ("GARCH11", "\\operatorname{GARCH11}")

    def update(self, node: Node):
        """Return the update mapping for the noise RV."""
        # Since noise is a shared variable it shows up as the last node input
        return {node.inputs[-1]: node.outputs[0]}


class GARCH11(Distribution):
    r"""
    GARCH(1,1) with Normal innovations. The model is specified by

    .. math::
        y_t \sim N(0, \sigma_t^2)

    .. math::
        \sigma_t^2 = \omega + \alpha_1 * y_{t-1}^2 + \beta_1 * \sigma_{t-1}^2

    where \sigma_t^2 (the error variance) follows a ARMA(1, 1) model.

    Parameters
    ----------
    omega: tensor
        omega > 0, mean variance
    alpha_1: tensor
        alpha_1 >= 0, autoregressive term coefficient
    beta_1: tensor
        beta_1 >= 0, alpha_1 + beta_1 < 1, moving average term coefficient
    initial_vol: tensor
        initial_vol >= 0, initial volatility, sigma_0
    """

    rv_type = GARCH11RV

    def __new__(cls, *args, steps=None, **kwargs):
        steps = get_steps(
            steps=steps,
            shape=None,  # Shape will be checked in `cls.dist`
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            step_shape_offset=1,
        )
        return super().__new__(cls, *args, steps=steps, **kwargs)

    @classmethod
    def dist(cls, omega, alpha_1, beta_1, initial_vol, *, steps=None, **kwargs):
        steps = get_steps(steps=steps, shape=kwargs.get("shape", None), step_shape_offset=1)
        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        steps = at.as_tensor_variable(intX(steps), ndim=0)

        omega = at.as_tensor_variable(omega)
        alpha_1 = at.as_tensor_variable(alpha_1)
        beta_1 = at.as_tensor_variable(beta_1)
        initial_vol = at.as_tensor_variable(initial_vol)

        init_dist = Normal.dist(0, initial_vol)
        # Tell Aeppl to ignore init_dist, as it will be accounted for in the logp term
        init_dist = ignore_logprob(init_dist)

        return super().dist([omega, alpha_1, beta_1, initial_vol, init_dist, steps], **kwargs)

    @classmethod
    def rv_op(cls, omega, alpha_1, beta_1, initial_vol, init_dist, steps, size=None):
        if size is not None:
            batch_size = size
        else:
            # In this case the size of the init_dist depends on the parameters shape
            batch_size = at.broadcast_shape(omega, alpha_1, beta_1, initial_vol)
        init_dist = change_dist_size(init_dist, batch_size)
        # initial_vol = initial_vol * at.ones(batch_size)

        # Create OpFromGraph representing random draws from GARCH11 process
        # Variables with underscore suffix are dummy inputs into the OpFromGraph
        init_ = init_dist.type()
        initial_vol_ = initial_vol.type()
        omega_ = omega.type()
        alpha_1_ = alpha_1.type()
        beta_1_ = beta_1.type()
        steps_ = steps.type()

        noise_rng = aesara.shared(np.random.default_rng())

        def step(prev_y, prev_sigma, omega, alpha_1, beta_1, rng):
            new_sigma = at.sqrt(
                omega + alpha_1 * at.square(prev_y) + beta_1 * at.square(prev_sigma)
            )
            next_rng, new_y = Normal.dist(mu=0, sigma=new_sigma, rng=rng).owner.outputs
            return (new_y, new_sigma), {rng: next_rng}

        (y_t, _), innov_updates_ = aesara.scan(
            fn=step,
            outputs_info=[init_, initial_vol_ * at.ones(batch_size)],
            non_sequences=[omega_, alpha_1_, beta_1_, noise_rng],
            n_steps=steps_,
            strict=True,
        )
        (noise_next_rng,) = tuple(innov_updates_.values())

        garch11_ = at.concatenate([init_[None, ...], y_t], axis=0).dimshuffle(
            tuple(range(1, y_t.ndim)) + (0,)
        )

        garch11_op = GARCH11RV(
            inputs=[omega_, alpha_1_, beta_1_, initial_vol_, init_, steps_],
            outputs=[noise_next_rng, garch11_],
            ndim_supp=1,
        )

        garch11 = garch11_op(omega, alpha_1, beta_1, initial_vol, init_dist, steps)
        return garch11


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
    value_dimswapped = value.dimshuffle((value.ndim - 1,) + tuple(range(0, value.ndim - 1)))
    initial_vol = initial_vol * at.ones_like(value_dimswapped[0])

    def volatility_update(x, vol, w, a, b):
        return at.sqrt(w + a * at.square(x) + b * at.square(vol))

    vol, _ = aesara.scan(
        fn=volatility_update,
        sequences=[value_dimswapped[:-1]],
        outputs_info=[initial_vol],
        non_sequences=[omega, alpha_1, beta_1],
        strict=True,
    )
    sigma_t = at.concatenate([[initial_vol], vol])
    # Compute and collapse logp across time dimension
    innov_logp = at.sum(logp(Normal.dist(0, sigma_t), value_dimswapped), axis=0)
    return innov_logp


@_moment.register(GARCH11RV)
def garch11_moment(op, rv, omega, alpha_1, beta_1, initial_vol, init_dist, steps, noise_rng):
    # GARCH(1,1) mean is zero
    return at.zeros_like(rv)


class EulerMaruyama(distribution.Continuous):
    r"""
    Stochastic differential equation discretized with the Euler-Maruyama method.

    Parameters
    ----------
    dt: float
        time step of discretization
    sde_fn: callable
        function returning the drift and diffusion coefficients of SDE
    sde_pars: tuple
        parameters of the SDE, passed as ``*args`` to ``sde_fn``
    """

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} has not yet been ported to PyMC 4.0.")

    @classmethod
    def dist(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} has not yet been ported to PyMC 4.0.")

    def __init__(self, dt, sde_fn, sde_pars, *args, **kwds):
        super().__init__(*args, **kwds)
        self.dt = dt = at.as_tensor_variable(dt)
        self.sde_fn = sde_fn
        self.sde_pars = sde_pars

    def logp(self, x):
        """
        Calculate log-probability of EulerMaruyama distribution at specified value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        xt = x[:-1]
        f, g = self.sde_fn(x[:-1], *self.sde_pars)
        mu = xt + self.dt * f
        sigma = at.sqrt(self.dt) * g
        return at.sum(Normal.dist(mu=mu, sigma=sigma).logp(x[1:]))

    def _distr_parameters_for_repr(self):
        return ["dt"]


class MvGaussianRandomWalk(distribution.Continuous):
    r"""
    Multivariate Random Walk with Normal innovations

    Parameters
    ----------
    mu: tensor
        innovation drift, defaults to 0.0
    cov: tensor
        pos def matrix, innovation covariance matrix
    tau: tensor
        pos def matrix, inverse covariance matrix
    chol: tensor
        Cholesky decomposition of covariance matrix
    init: distribution
        distribution for initial value (Defaults to Flat())

    Notes
    -----
    Only one of cov, tau or chol is required.

    """

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} has not yet been ported to PyMC 4.0.")

    @classmethod
    def dist(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} has not yet been ported to PyMC 4.0.")

    def __init__(
        self, mu=0.0, cov=None, tau=None, chol=None, lower=True, init=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.init = init or Flat.dist()
        self.innovArgs = (mu, cov, tau, chol, lower)
        self.innov = multivariate.MvNormal.dist(*self.innovArgs, shape=self.shape)
        self.mean = at.as_tensor_variable(0.0)

    def logp(self, x):
        """
        Calculate log-probability of Multivariate Gaussian
        Random Walk distribution at specified value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """

        if x.ndim == 1:
            x = x[np.newaxis, :]

        x_im1 = x[:-1]
        x_i = x[1:]

        return self.init.logp_sum(x[0]) + self.innov.logp_sum(x_i - x_im1)

    def _distr_parameters_for_repr(self):
        return ["mu", "cov"]

    def random(self, point=None, size=None):
        """
        Draw random values from MvGaussianRandomWalk.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int or tuple of ints, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array


        Examples
        -------

        .. code-block:: python

            with pm.Model():
                mu = np.array([1.0, 0.0])
                cov = np.array([[1.0, 0.0],
                                [0.0, 2.0]])

                # draw one sample from a 2-dimensional Gaussian random walk with 10 timesteps
                sample = MvGaussianRandomWalk(mu, cov, shape=(10, 2)).random()

                # draw three samples from a 2-dimensional Gaussian random walk with 10 timesteps
                sample = MvGaussianRandomWalk(mu, cov, shape=(10, 2)).random(size=3)

                # draw four samples from a 2-dimensional Gaussian random walk with 10 timesteps,
                # indexed with a (2, 2) array
                sample = MvGaussianRandomWalk(mu, cov, shape=(10, 2)).random(size=(2, 2))

        """

        # for each draw specified by the size input, we need to draw time_steps many
        # samples from MvNormal.

        size = to_tuple(size)
        multivariate_samples = self.innov.random(point=point, size=size)
        # this has shape (size, self.shape)
        if len(self.shape) == 2:
            # have time dimension in first slot of shape. Therefore the time
            # component can be accessed with the index equal to the length of size.
            time_axis = len(size)
            multivariate_samples = multivariate_samples.cumsum(axis=time_axis)
            if time_axis != 0:
                # this for loop covers the case where size is a tuple
                for idx in np.ndindex(size):
                    multivariate_samples[idx] = (
                        multivariate_samples[idx] - multivariate_samples[idx][0]
                    )
            else:
                # size was passed as None
                multivariate_samples = multivariate_samples - multivariate_samples[0]

        # if the above statement fails, then only a spatial dimension was passed in for self.shape.
        # Therefore don't subtract off the initial value since otherwise you get all zeros
        # as your output.
        return multivariate_samples


class MvStudentTRandomWalk(MvGaussianRandomWalk):
    r"""
    Multivariate Random Walk with StudentT innovations

    Parameters
    ----------
    nu: degrees of freedom
    mu: tensor
        innovation drift, defaults to 0.0
    cov: tensor
        pos def matrix, innovation covariance matrix
    tau: tensor
        pos def matrix, inverse covariance matrix
    chol: tensor
        Cholesky decomposition of covariance matrix
    init: distribution
        distribution for initial value (Defaults to Flat())
    """

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} has not yet been ported to PyMC 4.0.")

    @classmethod
    def dist(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} has not yet been ported to PyMC 4.0.")

    def __init__(self, nu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nu = at.as_tensor_variable(nu)
        self.innov = multivariate.MvStudentT.dist(self.nu, None, *self.innovArgs)

    def _distr_parameters_for_repr(self):
        return ["nu", "mu", "cov"]
