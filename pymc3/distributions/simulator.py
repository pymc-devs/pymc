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

import logging
import warnings

import aesara
import aesara.tensor as at
import numpy as np

from aesara.graph.op import Apply, Op
from aesara.graph.utils import MetaType
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable
from scipy.spatial import cKDTree

from pymc3.aesaraf import floatX
from pymc3.distributions.distribution import NoDistribution
from pymc3.distributions.logprob import _logp

__all__ = ["Simulator", "SimulatorRV"]

_log = logging.getLogger("pymc3")


def identity(x):
    """Identity function, used as a summary statistics."""
    return x


def gaussian(epsilon, obs_data, sim_data):
    """Gaussian kernel."""
    return -0.5 * ((obs_data - sim_data) / epsilon) ** 2


def laplace(epsilon, obs_data, sim_data):
    """Laplace kernel."""
    return -at.abs_((obs_data - sim_data) / epsilon)


class SimulatorRV(RandomVariable):
    """
    Base class for SimulatorRVs

    This should be subclassed when defining custom Simulator objects.

    The following class attributes must be defined:

    ndim_supp : int
        Number of dimensions of the SimulatorRV (0 for scalar, 1 for vector, etc.)
    ndims_params: list[int]
        Number of minimum dimensions of each parameter of the RV. For example,
        if the Simulator accepts two scalar inputs, it should be `[0, 0]`
    fn: callable
        Python random simulator function. Should expect the following signature
        (rng, arg1, arg2, ... argn, size), where rng is a numpy.random.RandomStream()
        and size defines the size of the desired sample. The epsilon parameter
        is not passed to the callable.

    The following class attributes can be optionally defined:

    distance : Aesara Op, callable or str
        Distance function. Available options are `"gaussian"` (default), `"laplacian"`,
        `"kullback_leibler"` or a user defined function (or Aesara Op) that takes
        epsilon, the summary statistics of observed_data and the summary statistics
        of simulated_data as input.
        ``gaussian`` :math: `-0.5 \\left(\\left(\frac{xo - xs}{\\epsilon}\right)^2\right)`
        ``laplace`` :math: `{\\left(\frac{|xo - xs|}{\\epsilon}\right)}`
        ``kullback_leibler`` `:math: d \\sum(-\\log(\frac{\nu_d} {\rho_d}) / \\epsilon) + log_r`
        gaussian + ``sum_stat="sort"`` is equivalent to the 1D 2-wasserstein distance
        laplace + ``sum_stat="sort"`` is equivalent to the the 1D 1-wasserstein distance

    sum_stat: Aesara Op, callable or str
        Summary statistic function. Available options are `"indentity"` (default),
        `"sort"`, `"mean"`, `"median"`. If a callable (or Aesara Op) is defined,
        it should return a 1d numpy array (or Aesara vector).

    epsilon: float or array
        Scaling parameter for the distance functions. It should be a float or
        an array of the same size of the output of ``sum_stat``. Defaults to ``1.0``

    Other class attributes used by `RandomVariables` can also be defined. See the
    documentation for aesara.tensor.random.op.RandomVariable for more information.

    Examples
    --------
    .. code-block:: python

        def my_simulator_fn(rng, loc, scale, size):
            return rng.normal(loc, scale, size=size)

        class MySimulatorRV(pm.SimulatorRV):
            ndim_supp = 0
            ndims_params = [0, 0]
            fn = my_simulator_fn
            distance = "gaussian"
            sum_stat = "sort"
            epsilon = 1.0

        my_simulator = MySimulatorRV()

        with pm.Model() as m:
            simulator = pm.Simulator("sim", my_simulator, 0, 1, observed=data)

    """

    name = "SimulatorRV"
    ndim_supp = None
    ndims_params = None
    dtype = "floatX"
    _print_name = ("Simulator", "\\operatorname{Simulator}")

    fn = None
    distance = gaussian
    sum_stat = identity
    epsilon = 1.0

    def __new__(cls, *args, **kwargs):
        if cls.fn is None:
            raise ValueError("SimulatorRV fn was not specified")

        if cls.ndim_supp is None:
            raise ValueError(
                "SimulatorRV must specify `ndim_supp`. This is the minimum"
                "number of dimensions for a single random draw from the simulator."
                "\nFor a univariate simulator `ndims_supp = 0`"
            )

        if cls.ndims_params is None:
            raise ValueError(
                "Simulator RV must specify `ndims_params`. This is the minimum"
                "number of dimensions for every parameter that the Simulator takes,"
                "including epsilon.\nFor a Simulator that can take two scalar "
                "parameters` ndims_params = [0, 0, 0]"
            )

        distance = cls.distance
        if not isinstance(distance, Op):
            if distance == "gaussian":
                distance = gaussian
            elif distance == "laplace":
                distance = laplace
            elif distance == "kullback_leibler":
                raise NotImplementedError("KL not refactored yet")
                # TODO: Wrap KL in aesara OP
                # distance = KullbackLiebler(observed)
                # if sum_stat != "identity":
                #     _log.info(f"Automatically setting sum_stat to identity as expected by {distance}")
                #     sum_stat = "identity"
            elif callable(distance):
                distance = create_distance_op_from_fn(distance)
            else:
                raise ValueError(f"The distance metric {distance} is not implemented")

        sum_stat = cls.sum_stat
        if not isinstance(sum_stat, Op):
            if sum_stat == "identity":
                sum_stat = identity
            elif sum_stat == "sort":
                sum_stat = at.sort
            elif sum_stat == "mean":
                sum_stat = at.mean
            elif sum_stat == "median":
                # Missing in Aesara, see aesara/issues/525
                sum_stat = create_sum_stat_op_from_fn(np.median)
            elif callable(sum_stat):
                sum_stat = create_sum_stat_op_from_fn(sum_stat)
            else:
                raise ValueError(f"The summary statistic {sum_stat} is not implemented")

        cls.distance = distance
        cls.sum_stat = sum_stat
        cls.epsilon = at.as_tensor_variable(floatX(cls.epsilon))
        return super().__new__(cls)

    @classmethod
    def rng_fn(cls, *args, **kwargs):
        return cls.fn(*args, **kwargs)

    @classmethod
    def _distance(cls, epsilon, value, simulated_value):
        return cls.distance(epsilon, value, simulated_value)

    @classmethod
    def _sum_stat(cls, value):
        return cls.sum_stat(value)


class Simulator(NoDistribution):
    r"""
    Simulator pseudo-distribution, used for Approximate Bayesian Inference (ABC)
    with Sequential Monte Carlo (SMC) sampling (i.e.,`pm.sample_smc`).

    The first argument should be an instance of a `pm.SimulatorRV` subclass that
    defines the Simulator object. See the documentation for `pm.SimulatorRV` for
    details on how to define a custom Simulator object

    Simulator distributions have a stochastic pseudo-loglikelihood defined by
    a distance metric between the observed data and simulated data, and tweaked
    by a hyper-parameter `epsilon`.

    Parameters
    ----------
    simulator: `pm.SimulatorRV` `Op`
        Simulator object. See `pm.SimulatorRV` docstrings for more details
    params: list
        Parameters used by the Simulator random function. Parameters can also
        be passed by order, in which case the keyword argumetn ``params`` is
        ignored.

    Examples
    --------
    .. code-block:: python

        def my_simulator_fn(rng, loc, scale, size):
            return rng.normal(loc, scale, size=size)

        class MySimulatorRV(pm.SimulatorRV):
            ndim_supp = 0
            ndims_params = [0, 0, 0]
            fn = my_simulator_fn
            distance = "gaussian"
            sum_stat = "sort"
            epsilon = 1.0

        my_simulator = MySimulatorRV()

        with pm.Model() as m:
            simulator = pm.Simulator("sim", my_simulator, 0, 1, observed=data)

    """

    def __new__(cls, name, simulator, *params, **kwargs):
        cls.check_simulator_args(simulator, *params, **kwargs)

        # Register custom op and logp
        cls.rv_op = simulator
        rv_type = type(simulator)
        NoDistribution.register(rv_type)

        @_logp.register(rv_type)
        def logp(op, sim_rv, rvs_to_values, *sim_params, **kwargs):
            value_var = rvs_to_values.get(sim_rv, sim_rv)
            return cls.logp(
                value_var,
                sim_rv,
                *sim_params,
            )

        return super().__new__(cls, name, simulator, *params, **kwargs)

    @classmethod
    def dist(cls, simulator, *params, **kwargs):
        cls.check_simulator_args(simulator, *params, **kwargs)

        # Compatibility with V3 params keyword argument
        if "params" in kwargs:
            warnings.warn(
                "The kwarg ``params`` will be deprecated. You should pass the "
                'Simulator parameters in order `pm.Simulator("sim", param1, param2, ...)',
                DeprecationWarning,
                stacklevel=2,
            )
            assert not params
            params = kwargs.pop("params")

        params = [at.as_tensor_variable(floatX(param)) for param in params]
        return super().dist([*params], **kwargs)

    @classmethod
    def logp(cls, value, sim_rv, *sim_params):
        # Create a new simulatorRV that is identical to the original one
        sim_op = sim_rv.owner.op
        sim_value = at.as_tensor_variable(sim_op.make_node(*sim_rv.owner.inputs))
        sim_value.name = "sim_value"

        # Override rng to avoid non-randomness in parallel sampling
        # TODO: Model rngs should be updated prior to multiprocessing split,
        #  in which case this would not be needed. However, that would have to be
        #  done for every sampler that may accomodate Simulators
        sim_value.owner.inputs[0].set_value(np.random.RandomState())

        return sim_op._distance(
            sim_op.epsilon,
            sim_op._sum_stat(value),
            sim_op._sum_stat(sim_value),
        )

    @classmethod
    def check_simulator_args(cls, simulator, *args, **kwargs):
        if not isinstance(simulator, SimulatorRV):
            if isinstance(simulator, MetaType):
                raise ValueError(
                    f"simulator {simulator} does not seem to be an instantiated "
                    f"class. Did you forget to call `{simulator}()`?"
                )
            raise ValueError(
                f"simulator {simulator} should be a subclass instance of "
                f"`pm.SimulatorRV` but got {type(simulator)}"
            )

        n_params = len(args) + len(kwargs.get("params", []))
        if n_params != len(simulator.ndims_params):
            raise ValueError(
                f"`Simulator` expected {len(simulator.ndims_params)} parameters"
                f"but got {n_params}."
            )

        if "distance" in kwargs:
            raise ValueError(
                "distance is no longer defined when calling `pm.Simulator`. It"
                "should be defined as a class attribute of the simulator object."
                "See pm.SimulatorRV for more details."
            )

        if "sum_stat" in kwargs:
            raise ValueError(
                "sum_stat is no longer defined when calling `pm.Simulator`. It"
                "should be defined as a class attribute of the simulator object."
                "See pm.SimulatorRV for more details."
            )

        if "epsilon" in kwargs:
            raise ValueError(
                "epsilon is no longer defined when calling `pm.Simulator`. It"
                "should be defined as a class attribute of the simulator object."
                "See pm.SimulatorRV for more details."
            )


class KullbackLiebler:
    """Approximate Kullback-Liebler."""

    def __init__(self, obs_data):
        if obs_data.ndim == 1:
            obs_data = obs_data[:, None]
        n, d = obs_data.shape
        rho_d, _ = cKDTree(obs_data).query(obs_data, 2)
        self.rho_d = rho_d[:, 1]
        self.d_n = d / n
        self.log_r = np.log(n / (n - 1))
        self.obs_data = obs_data

    def __call__(self, epsilon, obs_data, sim_data):
        if sim_data.ndim == 1:
            sim_data = sim_data[:, None]
        nu_d, _ = cKDTree(sim_data).query(self.obs_data, 1)
        return self.d_n * np.sum(-np.log(nu_d / self.rho_d) / epsilon) + self.log_r


scalarX = at.dscalar if aesara.config.floatX == "float64" else at.fscalar
vectorX = at.dvector if aesara.config.floatX == "float64" else at.fvector


def create_sum_stat_op_from_fn(fn):
    # Check if callable returns TensorVariable with dummy inputs
    try:
        res = fn(vectorX())
        if isinstance(res, TensorVariable):
            return fn
    except Exception:
        pass

    # Otherwise, automatically wrap in Aesara Op
    class SumStat(Op):
        def make_node(self, x):
            x = at.as_tensor_variable(x)
            return Apply(self, [x], [vectorX()])

        def perform(self, node, inputs, outputs):
            (x,) = inputs
            outputs[0][0] = np.atleast_1d(fn(x)).astype(aesara.config.floatX)

    return SumStat()


def create_distance_op_from_fn(fn):
    # Check if callable returns TensorVariable with dummy inputs
    try:
        res = fn(scalarX(), vectorX(), vectorX())
        if isinstance(res, TensorVariable):
            return fn
    except Exception:
        pass

    # Otherwise, automatically wrap in Aesara Op
    class Distance(Op):
        def make_node(self, epsilon, obs_data, sim_data):
            epsilon = at.as_tensor_variable(epsilon)
            obs_data = at.as_tensor_variable(obs_data)
            sim_data = at.as_tensor_variable(sim_data)
            return Apply(self, [epsilon, obs_data, sim_data], [vectorX()])

        def perform(self, node, inputs, outputs):
            eps, obs_data, sim_data = inputs
            outputs[0][0] = np.atleast_1d(fn(eps, obs_data, sim_data)).astype(aesara.config.floatX)

    return Distance()
