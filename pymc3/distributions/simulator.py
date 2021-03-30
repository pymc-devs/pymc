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

import numpy as np

from scipy.spatial import cKDTree

from pymc3.distributions.distribution import NoDistribution, draw_values, to_tuple

__all__ = ["Simulator"]

_log = logging.getLogger("pymc3")


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

    def __init__(
        self,
        function,
        *args,
        params=None,
        distance="gaussian",
        sum_stat="identity",
        epsilon=1,
        **kwargs,
    ):
        self.function = function
        self.params = params
        observed = self.data
        self.epsilon = epsilon

        if distance == "gaussian":
            self.distance = gaussian
        elif distance == "laplace":
            self.distance = laplace
        elif distance == "kullback_leibler":
            self.distance = KullbackLiebler(observed)
            if sum_stat != "identity":
                _log.info(f"Automatically setting sum_stat to identity as expected by {distance}")
                sum_stat = "identity"
        elif hasattr(distance, "__call__"):
            self.distance = distance
        else:
            raise ValueError(f"The distance metric {distance} is not implemented")

        if sum_stat == "identity":
            self.sum_stat = identity
        elif sum_stat == "sort":
            self.sum_stat = np.sort
        elif sum_stat == "mean":
            self.sum_stat = np.mean
        elif sum_stat == "median":
            self.sum_stat = np.median
        elif hasattr(sum_stat, "__call__"):
            self.sum_stat = sum_stat
        else:
            raise ValueError(f"The summary statistics {sum_stat} is not implemented")

        super().__init__(shape=np.prod(observed.shape), dtype=observed.dtype, *args, **kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from Simulator.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be conditioned (uses default
            point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not specified).

        Returns
        -------
        array
        """
        size = to_tuple(size)
        params = draw_values([*self.params], point=point, size=size)
        if len(size) == 0:
            return self.function(*params)
        else:
            return np.array([self.function(*params) for _ in range(size[0])])

    def _str_repr(self, name=None, dist=None, formatting="plain"):
        if dist is None:
            dist = self
        name = name
        function = dist.function.__name__
        params = ", ".join([var.name for var in dist.params])
        sum_stat = self.sum_stat.__name__ if hasattr(self.sum_stat, "__call__") else self.sum_stat
        distance = getattr(self.distance, "__name__", self.distance.__class__.__name__)

        if "latex" in formatting:
            return f"$\\text{{{name}}} \\sim  \\text{{Simulator}}(\\text{{{function}}}({params}), \\text{{{distance}}}, \\text{{{sum_stat}}})$"
        else:
            return f"{name} ~ Simulator({function}({params}), {distance}, {sum_stat})"


def identity(x):
    """Identity function, used as a summary statistics."""
    return x


def gaussian(epsilon, obs_data, sim_data):
    """Gaussian kernel."""
    return -0.5 * ((obs_data - sim_data) / epsilon) ** 2


def laplace(epsilon, obs_data, sim_data):
    """Laplace kernel."""
    return -np.abs((obs_data - sim_data) / epsilon)


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
