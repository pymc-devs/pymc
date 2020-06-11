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

import numpy as np
from .distribution import NoDistribution

__all__ = ["Simulator"]


class Simulator(NoDistribution):
    def __init__(self, function, *args, params=None, **kwargs):
        """
        This class stores a function defined by the user in python language.
        
        function: function
            Simulation function defined by the user.
        params: list
            Parameters passed to function.
        *args and **kwargs: 
            Arguments and keywords arguments that the function takes.
        """

        self.function = function
        self.params = params
        observed = self.data
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

        raise NotImplementedError("Not implemented yet")

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        name = r"\text{%s}" % name
        function = dist.function
        params = dist.parameters
        sum_stat = dist.sum_stat
        return r"${} \sim \text{{Simulator}}(\mathit{{function}}={},~\mathit{{parameters}}={},~\mathit{{summary statistics}}={})$".format(
            name, function, params, sum_stat
        )
