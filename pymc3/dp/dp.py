#   Copyright 2021 The PyMC Developers
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


import pymc3 as pm

from pymc3.distributions.distribution import Distribution
from pymc3.distributions.dist_math import bound

from pymc3.dp.util import StickBreakingWeights

from aesara import tensor as at

__all__ = [
    "DirichletProcess",
]


class DPBase:
    R"""
    Dirichlet Process Base class
    """

    def __init__(self, name, alpha, base_dist, K):

        # if not (isinstance(base_dist_obj, Distribution)):
        #     raise TypeError(
        #         f"Supplied base_dist_obj must be a Distribution"
        #         "but got {type(base_dist_obj)} instead."
        #     )
        self.name = name
        self.alpha = alpha
        self.base_dist = base_dist
        self.weights = StickBreakingWeights(name="sticks", alpha=self.alpha)
        self.K = K

    def __add__(self, other):
        return self.__class__(self.alpha, self.base_dist)

    def prior(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def marginal_likelihood(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def conditional(self, name, Xnew, *args, **kwargs):
        raise NotImplementedError


class DirichletProcess(DPBase):

    def __init__(self, name, alpha, base_dist, K):
        super().__init__(name, alpha, base_dist, K)

    def _build_prior(self, name, Xs, *args, **kwargs):
        alpha = pm.Gamma("dp-prior-alpha", 1, 1)

        dp = pm.Deterministic("dp", at.sum(at.mul(self.weights, at.lt(Xs, self.base_dist))))

        return dp

    def prior(self, name, Xs, **kwargs):
        # insert some checks for errors
        G = self._build_prior(name, Xs, **kwargs)
        self.Xs = Xs
        self.G = G

        return G
