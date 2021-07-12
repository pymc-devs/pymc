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

import warnings

sys.path.insert(0, "../../../pymc3")

import numpy as np

from aesara import tensor as at

import pymc3 as pm

import aesara
from aesara.tensor.random.op import RandomVariable
aesara.config.exception_verbosity="high"

# from pymc3.distributions.dist_math import bound
from pymc3.distributions.distribution import Distribution
# from pymc3.dp.util import StickBreakingWeights

__all__ = [
    "DirichletProcess",
]


def stick_breaking(betas):
    R"""
    betas: vector of Beta random variables
    """
    # need to make sure that betas is one-dimensional

    # if (betas.shape.eval()[0] <= 2):
    #     raise AttributeError(
    #         "More betas are needed to generate stick-breaking weights."
    #     )

    sticks = at.concatenate(
            [
                [1],
                at.cumprod(1 - betas[:-1]),
            ]
        )

    return betas*sticks

class DirichletProcess:
    R"""
    Dirichlet Process Base class
    """

    def __init__(self, name, alpha, base_dist, K=None):
        """
        Examples
        --------
        .. code:: python

            with pm.Model() as model:
                alpha = pm.Gamma("concentration", 1, 1)
                base_dist = pm.Normal("base_dist", 0, 3)
                dp = DirichletProcess("dp", alpha, base_dist, 20)
        """
        if isinstance(alpha, (int, float)):
            self.alpha = np.tile(alpha, reps=(K,))
        else:
            # at this point, self.alpha can be of length different than K, which is not okay
            if isinstance(alpha, (np.ndarray, aesara.graph.basic.Variable)):
                self.alpha = alpha
            else:
                raise ValueError(
                    "alpha parameter must be of type float, numpy.ndarray or "
                    f"pymc3.distributions.Distribution, but got {type(alpha)}"
                    "instead."
                )

        if not isinstance(base_dist, aesara.graph.basic.Variable):
            raise ValueError(
                "base_dist must be of type pymc3.distributions.Distribution"
                f"but got {type(base_dist)} instead."
            )

        if K is not None:
            try:
                if not isinstance(K, int):
                    assert K.is_integer()
            except (AssertionError, AttributeError):
                raise AttributeError(
                    "K need to be an int if specified."
                )

            try:
                base_dist_shape = base_dist.shape.eval()

                assert len(base_dist_shape) == 1
                assert base_dist_shape[0] == K
            except AssertionError:
                raise AttributeError(
                    f"The dimension of base_dist must be ({K},), "
                    f"but got {tuple(base_dist_shape)} instead"
                )

            if K < 30:
                # temporary, can think about raising a warning for too small K
                warnings.warn(
                    "You should specify K to be greater than 30."
                )

        else:
            K = 30
            # temporary, needs to be > 5*alpha + 2
            # raise error if not enough sticks

        self.name = name
        self.base_dist = base_dist
        self.K = K

        betas = pm.Beta("betas", 1., self.alpha)
        self.weights = pm.Deterministic("weights", stick_breaking(betas))

    def __add__(self, other):
        return self.__class__(self.alpha, self.base_dist)

    def __str__(self):
        return self.name

    @property
    def atoms(self):
        # to be overwritten in extensions of DirichletProcess?
        return self.base_dist

    def dp(self):
        pass


class DirichletProcessMixture(DirichletProcess):

    def __init__(self, name, alpha, base_dist, K):
        super().__init__(name, alpha, base_dist, K)


class DependentDensityRegression(DirichletProcess):

    def __init__(self, name, covariates, alpha=1, K=30, *args, **kwargs):
        # gaussian process for covariates?

        coeffs = pm.Normal("coeffs", 0, 10)
        super().__init__(name, alpha, pm.Normal, K)


if __name__ == "__main__":

    Xs = np.array([-1, 0, 1])
    Xnew = np.array([-3, -1, 0.5, 3.2, 4])
    K = 19

    with pm.Model() as model:
        try:
            Xs.shape[1]
        except IndexError as e:
            Xs = Xs[..., np.newaxis]

        try:
            Xnew.shape[1]
        except IndexError as e:
            Xnew = Xnew[..., np.newaxis]

        base_dist = pm.Normal("G0", 0, 3, shape=(K,))
        dp = DirichletProcess(
            name="dp",
            alpha=1.,
            base_dist=base_dist,
            K=K,
        )

        trace = pm.sample(
            draws=1000,
            chains=1,
        )

        print(trace.to_dict()["posterior"]["weights"][0].mean(axis=0))
