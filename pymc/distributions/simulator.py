#   Copyright 2024 - present The PyMC Developers
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

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.graph.op import Apply, Op
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.utils import safe_signature
from pytensor.tensor.variable import TensorVariable
from scipy.spatial import cKDTree

from pymc.distributions.distribution import Distribution, _support_point
from pymc.logprob.abstract import _logprob

__all__ = ["Simulator"]

_log = logging.getLogger(__name__)


class SimulatorRV(RandomVariable):
    """
    Base class for SimulatorRVs.

    This should be subclassed when defining custom Simulator objects.
    """

    name = "SimulatorRV"
    dtype = "floatX"
    _print_name = ("Simulator", "\\operatorname{Simulator}")

    fn = None
    _distance = None
    _sum_stat = None
    epsilon = None

    @classmethod
    def rng_fn(cls, *args, **kwargs):
        return cls.fn(*args, **kwargs)

    @classmethod
    def distance(cls, *args, **kwargs):
        return cls._distance(*args, **kwargs)

    @classmethod
    def sum_stat(cls, *args, **kwargs):
        return cls._sum_stat(*args, **kwargs)


class Simulator(Distribution):
    r"""
    Used for Approximate Bayesian Inference with SMC sampling via :func:`~pymc.sample_smc`.

    Simulator distributions have a stochastic pseudo-loglikelihood defined by
    a distance metric between the observed and simulated data, and tweaked
    by a hyper-parameter ``epsilon``.

    Parameters
    ----------
    fn : callable
        Python random simulator function. Should expect the following signature
        ``(rng, arg1, arg2, ... argn, size)``, where rng is a ``numpy.random.Generator``
        and ``size`` defines the size of the desired sample.
    *unnamed_params : list of TensorVariable
        Parameters used by the Simulator random function. Each parameter can be passed
        by order after fn, for example ``param1, param2, ..., paramN``. params can also
        be passed with keyword argument "params".
    params : list of TensorVariable
        Keyword form of ''unnamed_params''.
        One of unnamed_params or params must be provided.
        If passed both unnamed_params and params, an error is raised.
    distance : PyTensor_Op, callable or str, default "gaussian"
        Distance function. Available options are ``"gaussian"``, ``"laplace"``,
        ``"kullback_leibler"`` or a user defined function (or PyTensor_Op) that takes
        ``epsilon``, the summary statistics of observed_data and the summary statistics
        of simulated_data as input.

        ``gaussian``: :math:`-0.5 \left(\left(\frac{xo - xs}{\epsilon}\right)^2\right)`

        ``laplace``: :math:`-{\left(\frac{|xo - xs|}{\epsilon}\right)}`

        ``kullback_leibler``: :math:`\frac{d}{n} \frac{1}{\epsilon} \sum_i^n -\log \left( \frac{{\nu_d}_i}{{\rho_d}_i} \right) + \log_r` [1]_

        ``distance="gaussian"`` + ``sum_stat="sort"`` is equivalent to the 1D 2-wasserstein distance

        ``distance="laplace"`` + ``sum_stat="sort"`` is equivalent to the the 1D 1-wasserstein distance
    sum_stat : PyTensor_Op, callable or str, default "identity"
        Summary statistic function. Available options are ``"identity"``,
        ``"sort"``, ``"mean"``, ``"median"``. If a callable (or PyTensor_Op) is defined,
        it should return a 1d numpy array (or PyTensor vector).
    epsilon : tensor_like of float, default 1.0
        Scaling parameter for the distance functions. It should be a float or
        an array of the same size of the output of ``sum_stat``.
    ndim_supp : int, default 0
        Number of dimensions of the SimulatorRV (0 for scalar, 1 for vector, etc.)
    ndims_params : list of int, optional
        Number of minimum dimensions of each parameter of the RV. For example,
        if the Simulator accepts two scalar inputs, it should be ``[0, 0]``.
        Default to list of 0 with length equal to the number of parameters.
    class_name : str, optional
        Suffix name for the RandomVariable class which will wrap the Simulator methods.

    Examples
    --------
    .. code-block:: python

        def simulator_fn(rng, loc, scale, size):
            return rng.normal(loc, scale, size=size)


        with pm.Model() as m:
            loc = pm.Normal("loc", 0, 1)
            scale = pm.HalfNormal("scale", 1)
            simulator = pm.Simulator("simulator", simulator_fn, loc, scale, observed=data)
            idata = pm.sample_smc()

    References
    ----------
    .. [1] Pérez-Cruz, F. (2008, July). Kullback-Leibler divergence
        estimation of continuous distributions. In 2008 IEEE international
        symposium on information theory (pp. 1666-1670). IEEE.
        `link <https://ieeexplore.ieee.org/document/4595271>`__

    """

    rv_type = SimulatorRV

    def __new__(cls, name, *args, **kwargs):
        kwargs.setdefault("class_name", f"Simulator_{name}")
        return super().__new__(cls, name, *args, **kwargs)

    @classmethod
    def dist(  # type: ignore[override]
        cls,
        fn,
        *unnamed_params,
        params=None,
        distance="gaussian",
        sum_stat="identity",
        epsilon=1,
        signature=None,
        ndim_supp=None,
        ndims_params=None,
        dtype="floatX",
        class_name: str = "Simulator",
        **kwargs,
    ):
        if not isinstance(distance, Op):
            if distance == "gaussian":
                distance = gaussian
            elif distance == "laplace":
                distance = laplace
            elif distance == "kullback_leibler":
                raise NotImplementedError("KL not refactored yet")
                # TODO: Wrap KL in pytensor OP
                # distance = KullbackLeibler(observed)
                # if sum_stat != "identity":
                #     _log.info(f"Automatically setting sum_stat to identity as expected by {distance}")
                #     sum_stat = "identity"
            elif callable(distance):
                distance = create_distance_op_from_fn(distance)
            else:
                raise ValueError(f"The distance metric {distance} is not implemented")

        if not isinstance(sum_stat, Op):
            if sum_stat == "identity":
                sum_stat = identity
            elif sum_stat == "sort":
                sum_stat = pt.sort
            elif sum_stat == "mean":
                sum_stat = pt.mean
            elif sum_stat == "median":
                # Missing in PyTensor, see pytensor/issues/525
                sum_stat = create_sum_stat_op_from_fn(np.median)
            elif callable(sum_stat):
                sum_stat = create_sum_stat_op_from_fn(sum_stat)
            else:
                raise ValueError(f"The summary statistic {sum_stat} is not implemented")

        epsilon = pt.as_tensor_variable(epsilon)

        if params is None:
            params = unnamed_params
        else:
            if unnamed_params:
                raise ValueError("Cannot pass both unnamed parameters and `params`")

        if signature is None:
            # Assume scalar ndims_params
            temp_ndims_params = ndims_params if ndims_params is not None else [0] * len(params)
            # Assume scalar ndim_supp
            temp_ndim_supp = ndim_supp if ndim_supp is not None else 0
            signature = safe_signature(
                core_inputs_ndim=temp_ndims_params, core_outputs_ndim=[temp_ndim_supp]
            )

        return super().dist(
            params,
            fn=fn,
            signature=signature,
            ndim_supp=ndim_supp,
            ndims_params=ndims_params,
            dtype=dtype,
            distance=distance,
            sum_stat=sum_stat,
            epsilon=epsilon,
            class_name=class_name,
            **kwargs,
        )

    @classmethod
    def rv_op(
        cls,
        *params,
        fn,
        ndim_supp,
        ndims_params,
        dtype,
        distance,
        sum_stat,
        epsilon,
        class_name,
        signature,
        **kwargs,
    ):
        sim_op = type(
            class_name,
            (SimulatorRV,),
            {
                "name": class_name,
                "ndim_supp": ndim_supp,
                "ndims_params": ndims_params,
                "signature": signature,
                "dtype": dtype,
                "inplace": False,
                "fn": fn,
                "_distance": distance,
                "_sum_stat": sum_stat,
                "epsilon": epsilon,
            },
        )()
        return sim_op(*params, **kwargs)


@_support_point.register(SimulatorRV)
def simulator_support_point(op, rv, *inputs):
    sim_inputs = op.dist_params(rv.owner)
    # Take the mean of 10 draws
    multiple_sim = rv.owner.op(*sim_inputs, size=pt.concatenate([[10], rv.shape]))
    return pt.mean(multiple_sim, axis=0)


@_logprob.register(SimulatorRV)
def simulator_logp(op, values, *inputs, **kwargs):
    (value,) = values

    # Use a new rng to avoid non-randomness in parallel sampling
    # TODO: Model rngs should be updated prior to multiprocessing split,
    #  in which case this would not be needed. However, that would have to be
    #  done for every sampler that may accommodate Simulators
    rng = pytensor.shared(np.random.default_rng(), name="simulator_rng")
    # Create a new simulatorRV with identical inputs as the original one
    sim_value = op.make_node(rng, *inputs[1:]).default_output()
    sim_value.name = "simulator_value"

    return op.distance(
        op.epsilon,
        op.sum_stat(value),
        op.sum_stat(sim_value),
    )


def identity(x):
    """Identity function, used as a summary statistics."""
    return x


def gaussian(epsilon, obs_data, sim_data):
    """Gaussian kernel."""
    return -0.5 * ((obs_data - sim_data) / epsilon) ** 2


def laplace(epsilon, obs_data, sim_data):
    """Laplace kernel."""
    return -pt.abs((obs_data - sim_data) / epsilon)


class KullbackLeibler:
    """Approximate Kullback-Leibler."""

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


def create_sum_stat_op_from_fn(fn):
    vectorX = pt.dvector if pytensor.config.floatX == "float64" else pt.fvector

    # Check if callable returns TensorVariable with dummy inputs
    try:
        res = fn(vectorX())
        if isinstance(res, TensorVariable):
            return fn
    except Exception:
        pass

    # Otherwise, automatically wrap in PyTensor Op
    class SumStat(Op):
        def make_node(self, x):
            x = pt.as_tensor_variable(x)
            return Apply(self, [x], [vectorX()])

        def perform(self, node, inputs, outputs):
            (x,) = inputs
            outputs[0][0] = np.atleast_1d(fn(x)).astype(node.outputs[0].dtype)

    return SumStat()


def create_distance_op_from_fn(fn):
    scalarX = pt.dscalar if pytensor.config.floatX == "float64" else pt.fscalar
    vectorX = pt.dvector if pytensor.config.floatX == "float64" else pt.fvector

    # Check if callable returns TensorVariable with dummy inputs
    try:
        res = fn(scalarX(), vectorX(), vectorX())
        if isinstance(res, TensorVariable):
            return fn
    except Exception:
        pass

    # Otherwise, automatically wrap in PyTensor Op
    class Distance(Op):
        def make_node(self, epsilon, obs_data, sim_data):
            epsilon = pt.as_tensor_variable(epsilon)
            obs_data = pt.as_tensor_variable(obs_data)
            sim_data = pt.as_tensor_variable(sim_data)
            return Apply(self, [epsilon, obs_data, sim_data], [vectorX()])

        def perform(self, node, inputs, outputs):
            eps, obs_data, sim_data = inputs
            outputs[0][0] = np.atleast_1d(fn(eps, obs_data, sim_data)).astype(node.outputs[0].dtype)

    return Distance()
