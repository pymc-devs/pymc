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

import pytensor.tensor as pt

from pytensor.graph.replace import graph_replace

from pymc.pytensorf import floatX
from pymc.util import WithMemoization, locally_cachedmethod
from pymc.variational.opvi import node_property
from pymc.variational.test_functions import rbf

__all__ = ["Stein"]


class Stein(WithMemoization):
    def __init__(self, approx, kernel=rbf, use_histogram=True, temperature=1):
        self.approx = approx
        self.temperature = floatX(temperature)
        self._kernel_f = kernel
        self.use_histogram = use_histogram

    @property
    def input_joint_matrix(self):
        if self.use_histogram:
            return self.approx.joint_histogram
        else:
            return self.approx.symbolic_random

    @node_property
    def approx_symbolic_matrices(self):
        if self.use_histogram:
            return self.approx.collect("histogram")
        else:
            return self.approx.symbolic_randoms

    @node_property
    def dlogp(self):
        logp = self.logp_norm.sum()
        grad = pt.grad(logp, self.approx_symbolic_matrices)

        def flatten2(tensor):
            return tensor.flatten(2)

        return pt.concatenate(list(map(flatten2, grad)), -1)

    @node_property
    def grad(self):
        n = floatX(self.input_joint_matrix.shape[0])
        temperature = self.temperature
        svgd_grad = self.density_part_grad / temperature + self.repulsive_part_grad
        return svgd_grad / n

    @node_property
    def density_part_grad(self):
        Kxy = self.Kxy
        dlogpdx = self.dlogp
        return pt.dot(Kxy, dlogpdx)

    @node_property
    def repulsive_part_grad(self):
        t = self.approx.symbolic_normalizing_constant
        dxkxy = self.dxkxy
        return dxkxy / t

    @property
    def Kxy(self):
        return self._kernel()[0]

    @property
    def dxkxy(self):
        return self._kernel()[1]

    @node_property
    def logp_norm(self):
        sized_symbolic_logp = self.approx.sized_symbolic_logp
        if self.use_histogram:
            sized_symbolic_logp = graph_replace(
                sized_symbolic_logp,
                dict(zip(self.approx.symbolic_randoms, self.approx.collect("histogram"))),
                strict=False,
            )
        return sized_symbolic_logp / self.approx.symbolic_normalizing_constant

    @locally_cachedmethod
    def _kernel(self):
        return self._kernel_f(self.input_joint_matrix)
