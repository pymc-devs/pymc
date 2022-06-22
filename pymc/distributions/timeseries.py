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

from typing import Any, Optional, Tuple, Union

import aesara
import aesara.tensor as at
import numpy as np

from aeppl.abstract import MeasurableVariable, _get_measurable_outputs
from aeppl.logprob import _logprob
from aesara import scan
from aesara.compile.builders import OpFromGraph
from aesara.graph import FunctionGraph, optimize_graph
from aesara.graph.basic import Node
from aesara.raise_op import Assert
from aesara.tensor import TensorVariable
from aesara.tensor.basic_opt import ShapeFeature, topo_constant_folding
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.utils import normalize_size_param

from pymc.aesaraf import change_rv_size, convert_observed_data, floatX, intX
from pymc.distributions import distribution, multivariate
from pymc.distributions.continuous import Flat, Normal, get_tau_sigma
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import SymbolicDistribution, _moment, moment
from pymc.distributions.logprob import ignore_logprob, logp
from pymc.distributions.shape_utils import (
    Dims,
    Shape,
    convert_dims,
    rv_size_is_none,
    to_tuple,
)
from pymc.model import modelcontext
from pymc.util import check_dist_not_registered

__all__ = [
    "AR",
    "GaussianRandomWalk",
    "GARCH11",
    "EulerMaruyama",
    "MvGaussianRandomWalk",
    "MvStudentTRandomWalk",
]


def get_steps(
    steps: Optional[Union[int, np.ndarray, TensorVariable]],
    *,
    shape: Optional[Shape] = None,
    dims: Optional[Dims] = None,
    observed: Optional[Any] = None,
    step_shape_offset: int = 0,
):
    """Extract number of steps from shape / dims / observed information

    Parameters
    ----------
    steps:
        User specified steps for timeseries distribution
    shape:
        User specified shape for timeseries distribution
    dims:
        User specified dims for timeseries distribution
    observed:
        User specified observed data from timeseries distribution
    step_shape_offset:
        Difference between last shape dimension and number of steps in timeseries
        distribution, defaults to 0

    Returns
    -------
    steps
        Steps, if specified directly by user, or inferred from the last dimension of
        shape / dims / observed. When two sources of step information are provided,
        a symbolic Assert is added to ensure they are consistent.
    """
    inferred_steps = None
    if shape is not None:
        shape = to_tuple(shape)
        if shape[-1] is not ...:
            inferred_steps = shape[-1] - step_shape_offset

    if inferred_steps is None and dims is not None:
        dims = convert_dims(dims)
        if dims[-1] is not ...:
            model = modelcontext(None)
            inferred_steps = model.dim_lengths[dims[-1]] - step_shape_offset

    if inferred_steps is None and observed is not None:
        observed = convert_observed_data(observed)
        inferred_steps = observed.shape[-1] - step_shape_offset

    if inferred_steps is None:
        inferred_steps = steps
    # If there are two sources of information for the steps, assert they are consistent
    elif steps is not None:
        inferred_steps = Assert(msg="Steps do not match last shape dimension")(
            inferred_steps, at.eq(inferred_steps, steps)
        )
    return inferred_steps


class GaussianRandomWalkRV(RandomVariable):
    """
    GaussianRandomWalk Random Variable
    """

    name = "GaussianRandomWalk"
    ndim_supp = 1
    ndims_params = [0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("GaussianRandomWalk", "\\operatorname{GaussianRandomWalk}")

    def make_node(self, rng, size, dtype, mu, sigma, init_dist, steps):
        steps = at.as_tensor_variable(steps)
        if not steps.ndim == 0 or not steps.dtype.startswith("int"):
            raise ValueError("steps must be an integer scalar (ndim=0).")

        mu = at.as_tensor_variable(mu)
        sigma = at.as_tensor_variable(sigma)
        init_dist = at.as_tensor_variable(init_dist)

        # Resize init distribution
        size = normalize_size_param(size)
        # If not explicit, size is determined by the shapes of mu, sigma, and init
        init_dist_size = (
            size if not rv_size_is_none(size) else at.broadcast_shape(mu, sigma, init_dist)
        )
        init_dist = change_rv_size(init_dist, init_dist_size)

        return super().make_node(rng, size, dtype, mu, sigma, init_dist, steps)

    def _supp_shape_from_params(self, dist_params, reop_param_idx=0, param_shapes=None):
        steps = dist_params[3]

        return (steps + 1,)

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        mu: Union[np.ndarray, float],
        sigma: Union[np.ndarray, float],
        init_dist: Union[np.ndarray, float],
        steps: int,
        size: Tuple[int],
    ) -> np.ndarray:
        """Gaussian Random Walk generator.

        The init value is drawn from the Normal distribution with the same sigma as the
        innovations.

        Notes
        -----
        Currently does not support custom init distribution

        Parameters
        ----------
        rng: np.random.RandomState
           Numpy random number generator
        mu: array_like of float
           Random walk mean
        sigma: array_like of float
            Standard deviation of innovation (sigma > 0)
        init_dist: array_like of float
            Initialization value for GaussianRandomWalk
        steps: int
            Length of random walk, must be greater than 1. Returned array will be of size+1 to
            account as first value is initial value
        size: tuple of int
            The number of Random Walk time series generated

        Returns
        -------
        ndarray
        """

        if steps < 1:
            raise ValueError("Steps must be greater than 0")

        # If size is None then the returned series should be (*implied_dims, 1+steps)
        if size is None:
            # broadcast parameters with each other to find implied dims
            bcast_shape = np.broadcast_shapes(
                np.asarray(mu).shape,
                np.asarray(sigma).shape,
                np.asarray(init_dist).shape,
            )
            dist_shape = (*bcast_shape, int(steps))

        # If size is None then the returned series should be (*size, 1+steps)
        else:
            dist_shape = (*size, int(steps))

        # Add one dimension to the right, so that mu and sigma broadcast safely along
        # the steps dimension
        innovations = rng.normal(loc=mu[..., None], scale=sigma[..., None], size=dist_shape)
        grw = np.concatenate([init_dist[..., None], innovations], axis=-1)
        return np.cumsum(grw, axis=-1)


gaussianrandomwalk = GaussianRandomWalkRV()


class GaussianRandomWalk(distribution.Continuous):
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

    steps : tensor_like of int, optional
        Number of steps in Gaussian Random Walk (steps > 0). Only needed if shape is not
        provided.
    """

    rv_op = gaussianrandomwalk

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
    def dist(cls, mu=0.0, sigma=1.0, *, init_dist=None, steps=None, **kwargs) -> at.TensorVariable:

        mu = at.as_tensor_variable(floatX(mu))
        sigma = at.as_tensor_variable(floatX(sigma))

        steps = get_steps(
            steps=steps,
            shape=kwargs.get("shape"),
            step_shape_offset=1,
        )
        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        steps = at.as_tensor_variable(intX(steps))

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
        else:
            if not (
                isinstance(init_dist, at.TensorVariable)
                and init_dist.owner is not None
                and isinstance(init_dist.owner.op, RandomVariable)
                and init_dist.owner.op.ndim_supp == 0
            ):
                raise TypeError("init must be a univariate distribution variable")
            check_dist_not_registered(init_dist)

        # Ignores logprob of init var because that's accounted for in the logp method
        init_dist = ignore_logprob(init_dist)

        return super().dist([mu, sigma, init_dist, steps], **kwargs)

    def moment(rv, size, mu, sigma, init_dist, steps):
        grw_moment = at.zeros_like(rv)
        grw_moment = at.set_subtensor(grw_moment[..., 0], moment(init_dist))
        # Add one dimension to the right, so that mu broadcasts safely along the steps
        # dimension
        grw_moment = at.set_subtensor(grw_moment[..., 1:], mu[..., None])
        return at.cumsum(grw_moment, axis=-1)

    def logp(
        value: at.Variable,
        mu: at.Variable,
        sigma: at.Variable,
        init_dist: at.Variable,
        steps: at.Variable,
    ) -> at.TensorVariable:
        """Calculate log-probability of Gaussian Random Walk distribution at specified value."""

        # Calculate initialization logp
        init_logp = logp(init_dist, value[..., 0])

        # Make time series stationary around the mean value
        stationary_series = value[..., 1:] - value[..., :-1]
        # Add one dimension to the right, so that mu and sigma broadcast safely along
        # the steps dimension
        series_logp = logp(Normal.dist(mu[..., None], sigma[..., None]), stationary_series)

        return check_parameters(
            init_logp + series_logp.sum(axis=-1),
            steps > 0,
            msg="steps > 0",
        )


class AutoRegressiveRV(OpFromGraph):
    """A placeholder used to specify a log-likelihood for an AR sub-graph."""

    default_output = 1
    ar_order: int
    constant_term: bool

    def __init__(self, *args, ar_order, constant_term, **kwargs):
        self.ar_order = ar_order
        self.constant_term = constant_term
        super().__init__(*args, **kwargs)

    def update(self, node: Node):
        """Return the update mapping for the noise RV."""
        # Since noise is a shared variable it shows up as the last node input
        return {node.inputs[-1]: node.outputs[0]}


class AR(SymbolicDistribution):
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
    rho: tensor_like of float
        Tensor of autoregressive coefficients. The n-th entry in the last dimension is
        the coefficient for the n-th lag.
    sigma: tensor_like of float, optional
        Standard deviation of innovation (sigma > 0). Defaults to 1. Only required if
        tau is not specified.
    tau: tensor_like of float
        Precision of innovation (tau > 0).
    constant: bool, optional
        Whether the first element of rho should be used as a constant term in the AR
        process. Defaults to False
    init_dist: unnamed distribution
        Scalar or vector distribution for initial values. Distribution should be
        created via the `.dist()` API, and have shape (*shape[:-1], ar_order). If not,
        it will be automatically resized.

        .. warning:: init_dist will be cloned, rendering it independent of the one passed as input.

    ar_order: int, optional
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

    def __new__(cls, name, rho, *args, steps=None, constant=False, ar_order=None, **kwargs):
        rhos = at.atleast_1d(at.as_tensor_variable(floatX(rho)))
        ar_order = cls._get_ar_order(rhos=rhos, constant=constant, ar_order=ar_order)
        steps = get_steps(
            steps=steps,
            shape=None,  # Shape will be checked in `cls.dist`
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            step_shape_offset=ar_order,
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
        steps = get_steps(steps=steps, shape=kwargs.get("shape", None), step_shape_offset=ar_order)
        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        steps = at.as_tensor_variable(intX(steps), ndim=0)

        if init_dist is not None:
            if not isinstance(init_dist, TensorVariable) or not isinstance(
                init_dist.owner.op, RandomVariable
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
            shape_fg = FunctionGraph(
                outputs=[rhos.shape[-1]],
                features=[ShapeFeature()],
                clone=True,
            )
            (folded_shape,) = optimize_graph(shape_fg, custom_opt=topo_constant_folding).outputs
            folded_shape = getattr(folded_shape, "data", None)
            if folded_shape is None:
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
            batch_size = at.broadcast_shape(sigma, rhos[..., 0], init_dist[..., 0])
        if init_dist.owner.op.ndim_supp == 0:
            init_dist_size = (*batch_size, ar_order)
        else:
            # In this case the support dimension must cover for ar_order
            init_dist_size = batch_size
        init_dist = change_rv_size(init_dist, init_dist_size)

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
            inline=True,
        )

        ar = ar_op(rhos, sigma, init_dist, steps)
        return ar

    @classmethod
    def change_size(cls, rv, new_size, expand=False):

        if expand:
            old_size = rv.shape[:-1]
            new_size = at.concatenate([new_size, old_size])

        op = rv.owner.op
        return cls.rv_op(
            *rv.owner.inputs,
            ar_order=op.ar_order,
            constant_term=op.constant_term,
            size=new_size,
        )


MeasurableVariable.register(AutoRegressiveRV)


@_get_measurable_outputs.register(AutoRegressiveRV)
def _get_measurable_outputs_ar(op, node):
    # This tells Aeppl that the second output is the measurable one
    return [node.outputs[1]]


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


class GARCH11(distribution.Continuous):
    r"""
    GARCH(1,1) with Normal innovations. The model is specified by

    .. math::
        y_t = \sigma_t * z_t

    .. math::
        \sigma_t^2 = \omega + \alpha_1 * y_{t-1}^2 + \beta_1 * \sigma_{t-1}^2

    with z_t iid and Normal with mean zero and unit standard deviation.

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

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} has not yet been ported to PyMC 4.0.")

    @classmethod
    def dist(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} has not yet been ported to PyMC 4.0.")

    def __init__(self, omega, alpha_1, beta_1, initial_vol, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.omega = omega = at.as_tensor_variable(omega)
        self.alpha_1 = alpha_1 = at.as_tensor_variable(alpha_1)
        self.beta_1 = beta_1 = at.as_tensor_variable(beta_1)
        self.initial_vol = at.as_tensor_variable(initial_vol)
        self.mean = at.as_tensor_variable(0.0)

    def get_volatility(self, x):
        x = x[:-1]

        def volatility_update(x, vol, w, a, b):
            return at.sqrt(w + a * at.square(x) + b * at.square(vol))

        vol, _ = scan(
            fn=volatility_update,
            sequences=[x],
            outputs_info=[self.initial_vol],
            non_sequences=[self.omega, self.alpha_1, self.beta_1],
        )
        return at.concatenate([[self.initial_vol], vol])

    def logp(self, x):
        """
        Calculate log-probability of GARCH(1, 1) distribution at specified value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        vol = self.get_volatility(x)
        return at.sum(Normal.dist(0.0, sigma=vol).logp(x))

    def _distr_parameters_for_repr(self):
        return ["omega", "alpha_1", "beta_1"]


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
