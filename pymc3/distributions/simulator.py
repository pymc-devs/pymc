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
from .distribution import NoDistribution, draw_values

__all__ = ["Simulator"]

_log = logging.getLogger("pymc3")


class Simulator(NoDistribution):
    def __init__(
        self,
        function,
        *args,
        params=None,
        distance="gaussian_kernel",
        sum_stat="identity",
        epsilon=1,
        **kwargs,
    ):
        """
        This class stores a function defined by the user in Python language.
        
        function: function
            Python function defined by the user.
        params: list
            Parameters passed to function.
        distance: str or callable
            Distance functions. Available options are "gaussian_kernel" (default), "wasserstein",
            "energy" or a user defined function that takes epsilon (a scalar), and the summary
            statistics of observed_data, and simulated_data as input.
            ``gaussian_kernel`` :math: `\sum \left(-0.5  \left(\frac{xo - xs}{\epsilon}\right)^2\right)`
            ``wasserstein`` :math: `\frac{1}{n} \sum{\left(\frac{|xo - xs|}{\epsilon}\right)}`
            ``energy`` :math: `\sqrt{2} \sqrt{\frac{1}{n} \sum \left(\frac{|xo - xs|}{\epsilon}\right)^2}`
            For the wasserstein and energy distances the observed data xo and simulated data xs
            are internally sorted (i.e. the sum_stat is "sort").
        sum_stat: str or callable
            Summary statistics. Available options are ``indentity``, ``sort``, ``mean``, ``median``.
            If a callable is based it should return a number or a 1d numpy array.
        epsilon: float
            Standard deviation of the gaussian_kernel.
        *args and **kwargs: 
            Arguments and keywords arguments that the function takes.
        """

        self.function = function
        self.params = params
        observed = self.data
        self.epsilon = epsilon

        if distance == "gaussian_kernel":
            self.distance = gaussian_kernel
        elif distance == "wasserstein":
            self.distance = wasserstein
            if sum_stat != "sort":
                _log.info(f"Automatically setting sum_stat to sort as expected by {distance}")
                sum_stat = "sort"
        elif distance == "energy":
            self.distance = energy
            if sum_stat != "sort":
                _log.info(f"Automatically setting sum_stat to sort as expected by {distance}")
                sum_stat = "sort"
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
        Draw random values from Simulator
        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).
        Returns
        -------
        array
        """
        params = draw_values([*self.params], point=point, size=size)
        if size is None:
            return self.function(*params)
        else:
            return np.array([self.function(*params) for _ in range(size)])

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        name = name
        function = dist.function.__name__
        params = ", ".join([var.name for var in dist.params])
        sum_stat = self.sum_stat.__name__ if hasattr(self.sum_stat, "__call__") else self.sum_stat
        distance = self.distance.__name__
        return f"$\\text{{{name}}} \sim  \\text{{Simulator}}(\\text{{{function}}}({params}), \\text{{{distance}}}, \\text{{{sum_stat}}})$"


def identity(x):
    """Identity function, used as a summary statistics."""
    return x


def gaussian_kernel(epsilon, obs_data, sim_data):
    """gaussian distance function"""
    return np.sum(-0.5 * ((obs_data - sim_data) / epsilon) ** 2)


def wasserstein(epsilon, obs_data, sim_data):
    """Wasserstein distance function.
    
    We are assuming obs_data and sim_data are already sorted!
    """
    return np.mean(np.abs((obs_data - sim_data) / epsilon))


def energy(epsilon, obs_data, sim_data):
    """Energy distance function.
    
    We are assuming obs_data and sim_data are already sorted!
    """
    return 1.4142 * np.mean(((obs_data - sim_data) / epsilon) ** 2) ** 0.5
