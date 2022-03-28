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

import aesara
import aesara.tensor as at
import numpy as np

from aeppl.logprob import _logprob
from aesara.graph.op import Apply, Op
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable
from scipy.spatial import cKDTree

from pymc.aesaraf import floatX
from pymc.distributions.distribution import NoDistribution, _moment

__all__ = ["Simulator"]

_log = logging.getLogger("pymc")


class SimulatorRV(RandomVariable):
    """
    Base class for SimulatorRVs

    This should be subclassed when defining custom Simulator objects.
    """

    name = "SimulatorRV"
    ndim_supp = None
    ndims_params = None
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


class Simulator(NoDistribution):
    r"""
    Simulator distribution, used for Approximate Bayesian Inference (ABC)
    with Sequential Monte Carlo (SMC) sampling via ``pm.sample_smc``.

    Simulator distributions have a stochastic pseudo-loglikelihood defined by
    a distance metric between the observed and simulated data, and tweaked
    by a hyper-parameter ``epsilon``.

    Parameters
    ----------
    fn: callable
        Python random simulator function. Should expect the following signature
        ``(rng, arg1, arg2, ... argn, size)``, where rng is a ``numpy.random.RandomStream()``
        and ``size`` defines the size of the desired sample.
    params: list
        Parameters used by the Simulator random function. Parameters can also
        be passed by order, in which case the keyword argument ``params`` is
        ignored. Alternatively, each parameter can be passed by order after fn,
        ``param1, param2, ..., paramN``
    distance : Aesara Op, callable or str
        Distance function. Available options are ``"gaussian"`` (default), ``"laplace"``,
        ``"kullback_leibler"`` or a user defined function (or Aesara Op) that takes
        ``epsilon``, the summary statistics of observed_data and the summary statistics
        of simulated_data as input.

        ``gaussian``: :math:`-0.5 \left(\left(\frac{xo - xs}{\epsilon}\right)^2\right)`

        ``laplace``: :math:`-{\left(\frac{|xo - xs|}{\epsilon}\right)}`

        ``kullback_leibler``: :math:`\frac{d}{n} \frac{1}{\epsilon} \sum_i^n -\log \left( \frac{{\nu_d}_i}{{\rho_d}_i} \right) + \log_r` [1]_

        ``distance="gaussian"`` + ``sum_stat="sort"`` is equivalent to the 1D 2-wasserstein distance

        ``distance="laplace"`` + ``sum_stat="sort"`` is equivalent to the the 1D 1-wasserstein distance
    sum_stat: Aesara Op, callable or str
        Summary statistic function. Available options are ``"indentity"`` (default),
        ``"sort"``, ``"mean"``, ``"median"``. If a callable (or Aesara Op) is defined,
        it should return a 1d numpy array (or Aesara vector).
    epsilon: float or array
        Scaling parameter for the distance functions. It should be a float or
        an array of the same size of the output of ``sum_stat``. Defaults to ``1.0``
    ndim_supp : int
        Number of dimensions of the SimulatorRV (0 for scalar, 1 for vector, etc.)
        Defaults to ``0``.
    ndims_params: list[int]
        Number of minimum dimensions of each parameter of the RV. For example,
        if the Simulator accepts two scalar inputs, it should be ``[0, 0]``.
        Defaults to ``0`` for each parameter.

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

    def __new__(
        cls,
        name,
        fn,
        *unnamed_params,
        params=None,
        distance="gaussian",
        sum_stat="identity",
        epsilon=1,
        ndim_supp=0,
        ndims_params=None,
        dtype="floatX",
        **kwargs,
    ):

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

        epsilon = at.as_tensor_variable(floatX(epsilon))

        if params is None:
            params = unnamed_params
        else:
            if unnamed_params:
                raise ValueError("Cannot pass both unnamed parameters and `params`")

        # Assume scalar ndims_params
        if ndims_params is None:
            ndims_params = [0] * len(params)

        sim_op = type(
            f"Simulator_{name}",
            (SimulatorRV,),
            dict(
                name="Simulator",
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                dtype=dtype,
                inplace=False,
                fn=fn,
                _distance=distance,
                _sum_stat=sum_stat,
                epsilon=epsilon,
            ),
        )()

        # The logp function is registered to the more general SimulatorRV,
        # in order to avoid issues with multiprocessing / pickling

        # rv_type = type(sim_op)
        # NoDistribution.register(rv_type)
        NoDistribution.register(SimulatorRV)

        @_logprob.register(SimulatorRV)
        def logp(op, value_var_list, *dist_params, **kwargs):
            _dist_params = dist_params[3:]
            value_var = value_var_list[0]
            return cls.logp(value_var, op, dist_params)

        @_moment.register(SimulatorRV)
        def moment(op, rv, rng, size, dtype, *rv_inputs):
            return cls.moment(rv, *rv_inputs)

        cls.rv_op = sim_op
        return super().__new__(cls, name, *params, **kwargs)

    @classmethod
    def dist(cls, *params, **kwargs):
        return super().dist(params, **kwargs)

    @classmethod
    def moment(cls, rv, *sim_inputs):
        # Take the mean of 10 draws
        multiple_sim = rv.owner.op(*sim_inputs, size=at.concatenate([[10], rv.shape]))
        return at.mean(multiple_sim, axis=0)

    @classmethod
    def logp(cls, value, sim_op, sim_inputs):
        # Use a new rng to avoid non-randomness in parallel sampling
        # TODO: Model rngs should be updated prior to multiprocessing split,
        #  in which case this would not be needed. However, that would have to be
        #  done for every sampler that may accomodate Simulators
        rng = aesara.shared(np.random.default_rng())
        rng.tag.is_rng = True

        # Create a new simulatorRV with identical inputs as the original one
        sim_value = sim_op.make_node(rng, *sim_inputs[1:]).default_output()
        sim_value.name = "sim_value"

        return sim_op.distance(
            sim_op.epsilon,
            sim_op.sum_stat(value),
            sim_op.sum_stat(sim_value),
        )


def identity(x):
    """Identity function, used as a summary statistics."""
    return x


def gaussian(epsilon, obs_data, sim_data):
    """Gaussian kernel."""
    return -0.5 * ((obs_data - sim_data) / epsilon) ** 2


def laplace(epsilon, obs_data, sim_data):
    """Laplace kernel."""
    return -at.abs_((obs_data - sim_data) / epsilon)


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


def create_sum_stat_op_from_fn(fn):
    vectorX = at.dvector if aesara.config.floatX == "float64" else at.fvector

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
    scalarX = at.dscalar if aesara.config.floatX == "float64" else at.fscalar
    vectorX = at.dvector if aesara.config.floatX == "float64" else at.fvector

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
