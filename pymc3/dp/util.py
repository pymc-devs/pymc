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
from pymc3.distributions.dist_math import bound, normal_lcdf
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

    def __call__(self, alpha, size=None, **kwargs):
        return super().__call__(alpha, size=size, **kwargs)

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


class LarryNormalRV(RandomVariable):
    name = "normal"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Larry's normal", "\\operatorname{LarryNormal}")

    # @classmethod
    # def rng_fn(cls, rng, loc, scale, size):
    #     return np.random.normal(loc=loc, scale=scale, rng=rng, size=size)

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        return super().__call__(loc, scale, size=size, **kwargs)


larrynormal = LarryNormalRV()


class LarryNormal(Continuous):
    rv_op = larrynormal

    @classmethod
    def dist(cls, mu, sigma, **kwargs):
        return super().dist([mu, sigma], **kwargs)

    def logp(value, mu, sigma):
        return (-at.log(2*np.pi*sigma**2) - (value - mu)**2/sigma**2)/2

    def logcdf(value, mu, sigma):
        return bound(
            normal_lcdf(mu, sigma, value),
            0 < sigma,
        )


if __name__ == "__main__":

    with pm.Model() as model:
        sbw = StickBreakingWeights("test-sticks", alpha=1)
        # larry_norm = LarryNormal(name="larrynormal", mu=1, sigma=2)

        trace = pm.sample(1000)
        print(trace.to_dict()["posterior"]["test-sticks"][0].mean())
