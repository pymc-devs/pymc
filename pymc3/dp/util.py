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
import sys

sys.path.insert(0, "../../../pymc3")

import numpy as np

from aesara import tensor as at
from aesara.tensor.random.op import RandomVariable

import pymc3 as pm

from pymc3.distributions.continuous import assert_negative_support
from pymc3.distributions.dist_math import bound
from pymc3.distributions.distribution import Continuous

__all__ = [
    "StickBreakingWeights",
]


class StickBreakingWeightsRV(RandomVariable):
    name = "stick_breaking_weights"
    ndim_supp = 1
    ndims_params = [0]
    dtype = "floatX"
    _print_name = ("Stick-Breaking weights", "\\operatorname{StickBreakingWeights}")

    @classmethod
    def rng_fn(cls, rng, alpha, size):
        betas = rng.beta(1, alpha, size=size)
        sticks = np.concatenate(
            [
                [1],
                np.cumprod(1 - betas[:-1]),
            ]
        )

        return betas * sticks


stickbreakingweights = StickBreakingWeightsRV()


class StickBreakingWeights(Continuous):
    rv_op = stickbreakingweights

    @classmethod
    def dist(cls, alpha, *args, **kwargs):
        alpha = at.as_tensor_variable(alpha)

        assert_negative_support(alpha, "alpha", "StickBreakingWeights")

        return super().dist([alpha], **kwargs)

    def logp(value, alpha):
        return bound(
            at.sum(pm.Beta.logp(value, alpha=1, beta=alpha)),
            alpha > 0,
        )

    def _distr_parameters_for_repr(self):
        return ["alpha"]


if __name__ == "__main__":

    with pm.Model() as model:
        # sbw = StickBreakingWeights("test-sticks", alpha=1)
        sbw = pm.Dirichlet(
            name="sticks",
            a=np.ones(
                20,
            ),
        )
        trace = pm.sample(1000)
