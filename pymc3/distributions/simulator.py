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

import aesara.tensor as at
import numpy as np

from aesara.tensor.random.op import RandomVariable
from scipy.spatial import cKDTree

from pymc3.distributions.distribution import NoDistribution
from pymc3.distributions.logp import _logp

__all__ = ["Simulator"]

_log = logging.getLogger("pymc3")


class SimulatorRV(RandomVariable):
    """A placeholder for Simulator RVs"""

    _print_name = ("Simulator", "\\operatorname{Simulator}")
    fn = None

    @classmethod
    def rng_fn(cls, *args, **kwargs):
        return cls.fn(*args, **kwargs)


class Simulator(NoDistribution):
    r"""
    Define a simulator, from a Python function, to be used in ABC methods.

    Parameters
    ----------
    function: callable
        Python function defined by the user.
    params: list
        Parameters passed to function.
    distance: str or callable
        Distance functions. Available options are "gaussian" (default), "laplacian",
        "kullback_leibler" or a user defined function that takes epsilon, the summary statistics of
        observed_data and the summary statistics of simulated_data as input.
        ``gaussian`` :math: `-0.5 \left(\left(\frac{xo - xs}{\epsilon}\right)^2\right)`
        ``laplace`` :math: `{\left(\frac{|xo - xs|}{\epsilon}\right)}`
        ``kullback_leibler`` `:math: d \sum(-\log(\frac{\nu_d} {\rho_d}) / \epsilon) + log_r`
        gaussian + ``sum_stat="sort"`` is equivalent to the 1D 2-wasserstein distance
        laplace + ``sum_stat="sort"`` is equivalent to the the 1D 1-wasserstein distance
    sum_stat: str or callable
        Summary statistics. Available options are ``indentity``, ``sort``, ``mean``, ``median``.
        If a callable is based it should return a number or a 1d numpy array.
    epsilon: float or array
        Scaling parameter for the distance functions. It should be a float or an array of the
        same size of the output of ``sum_stat``.
    *args and **kwargs:
        Arguments and keywords arguments that the function takes.
    """

    def __new__(
        cls,
        name,
        fn,
        *,
        params=None,
        distance="gaussian",
        sum_stat="identity",
        epsilon=1,
        observed=None,
        ndim_supp=0,
        ndims_params=None,
        dtype="floatX",
        **kwargs,
    ):

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
        elif hasattr(distance, "__call__"):
            # TODO: Wrap (optionally) non symbolic distance in Aesara OP
            distance = distance
        else:
            raise ValueError(f"The distance metric {distance} is not implemented")

        if sum_stat == "identity":
            sum_stat = identity
        elif sum_stat == "sort":
            sum_stat = at.sort
        elif sum_stat == "mean":
            sum_stat = at.mean
        elif sum_stat == "median":
            sum_stat = at.median
        elif hasattr(sum_stat, "__call__"):
            # TODO: Wrap (optionally) non symbolic sum_stat in Aesara OP
            sum_stat = sum_stat
        else:
            raise ValueError(f"The summary statistics {sum_stat} is not implemented")

        if params is None:
            params = []

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
                # Specifc to Simulator
                fn=fn,
                # distance=distance,
                # sum_stat=sum_stat,
                # epsilon=epsilon,
            ),
        )()

        # Register custom logp
        rv_type = type(sim_op)

        @_logp.register(rv_type)
        def logp(op, sim_rv, rvs_to_values, *sim_params, **kwargs):
            value_var = rvs_to_values.get(sim_rv, sim_rv)
            return cls.logp(
                value_var,
                sim_rv,
                distance,
                sum_stat,
                epsilon,
            )

        cls.rv_op = sim_op
        return super().__new__(cls, name, params, observed=observed, **kwargs)

    @classmethod
    def logp(cls, value, sim_rv, distance, sum_stat, epsilon):
        # Create a new simulatorRV identically to the original one
        sim_op = sim_rv.owner.op
        sim_data = at.as_tensor_variable(sim_op.make_node(*sim_rv.owner.inputs))
        sim_data.name = "sim_data"

        return distance(epsilon, sum_stat(value), sum_stat(sim_data))


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
