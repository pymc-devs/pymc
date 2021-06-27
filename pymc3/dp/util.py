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
import numpy as np

import pymc3 as pm

from pymc3.distributions.distribution import Distribution
from pymc3.distributions.dist_math import bound
from pymc3.distributions.continuous import assert_negative_support

from aesara import tensor as at
from aesara.tensor.random.op import RandomVariable

__all__ = [
    "StickBreakingWeights",
]


class StickBreakingWeightsRV(RandomVariable):
    name = "stick_breaking_weights"
    ndim_supp = 1
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = (
        "Stick-Breaking weights",
        "\\operatorname{StickBreakingWeights}"
    )

    @classmethod
    def rng_fn(cls, rng, alpha, size=None):
        betas = rng.beta(1, alpha, size=size)
        sticks = np.concatenate(
            [
                [1],
                np.cumprod(1 - betas[:-1]),
            ]
        )

        return betas*sticks


stickbreakingweights = StickBreakingWeightsRV()


class StickBreakingWeights(Distribution):

    rv_op = stickbreakingweights

    # def __init__(self, alpha=1, K=30, **kwargs):
    #     self.alpha = alpha
    #     self.K = K
    #
    #     self._initialize_weights()
    #
    #     return self.weights
    #
    # def _initialize_weights(self):
    #     betas = pm.Beta(
    #         "beta-for-weights",
    #         alpha=1,
    #         beta=self.alpha,
    #         shape=(self.K,),
    #     )
    #     sticks = at.concatenate([[1], (1 - betas[:-1])])
    #
    #     self.betas = betas
    #     self.weights = pm.Deterministic(
    #         name="stick-breaking-weights",
    #         var=at.mul(betas, at.cumprod(sticks)),
    #     )
    #
    # @property
    # def weights(self):
    #     self._initialize_weights()

    @classmethod
    def dist(cls, alpha=1, K=30, *args, **kwargs):
        alpha = at.as_tensor_variable(alpha)

        assert_negative_support(alpha, "alpha", "StickBreakingWeights")

        return super().dist([alpha, K], **kwargs)

    def logp(value, alpha):
        return bound(
            at.sum(pm.Beta.logp(value, alpha=1, beta=alpha)),
            alpha > 0,
            broadcast_conditions=False,
        )
