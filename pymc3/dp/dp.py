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

import pymc3 as pm

# from pymc3.distributions.dist_math import bound
# from pymc3.distributions.distribution import Distribution
# from pymc3.dp.util import StickBreakingWeights

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

        if type(alpha) is int:
            self.alpha = np.tile(alpha, reps=(K,))
        else:
            self.alpha = alpha

        self.base_dist = base_dist
        # self.weights = StickBreakingWeights(name="sticks", alpha=self.alpha, size=(K,))
        self.weights = pm.Dirichlet(
            name="sticks",
            a=self.alpha,
        )
        self.K = K

    def __add__(self, other):
        return self.__class__(self.alpha, self.base_dist)

    def prior(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def marginal_likelihood(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def posterior(self, name, Xnew, *args, **kwargs):
        raise NotImplementedError


class DirichletProcess(DPBase):
    """
    Similarly to GPs, need to build prior before conditional?
    """

    def __init__(self, name, alpha, base_dist, K):
        super().__init__(name, alpha, base_dist, K)

    def _build_prior(self, name, Xs, *args, **kwargs):
        # if len(Xs.shape) == 1:
        #     self.Xs = Xs[..., np.newaxis]
        # else:
        self.Xs = Xs

        dirac = at.lt(self.base_dist, self.Xs)
        linear_comb = at.sum(at.mul(self.weights, dirac), axis=1)

        dp_prior = pm.Deterministic(
            name=name+"-dp-prior",
            var=linear_comb,
        )

        return dp_prior

    def prior(self, name, Xs, **kwargs):
        # insert some checks for errors

        G0 = self._build_prior(name, Xs, **kwargs)

        return G0

    def _build_posterior(self, name, Xs, Xnew=None):
        if name is None:
            name = "dp-post"

        Xs = at.atleast_1d(Xs)

        self._build_prior(name, Xs)

        if Xnew is None:
            Xnew = self.Xs

        dirac = at.sum(at.lt(Xnew, self.Xs), axis=1)

        empirical_cdf = at.lt(self.base_dist + dirac, Xnew) # P(X \leq x) since X is a distribution
        linear_comb = at.sum(at.mul(self.weights, empirical_cdf), axis=1)
        posterior_dp = pm.Deterministic(
            name=name,
            var=linear_comb,
        )

        return posterior_dp

    def posterior(self, name, Xs, Xnew=None, **kwargs):
        G_post = self._build_posterior(name, Xs, Xnew, **kwargs)

        return G_post


class DirichletProcessMixture(DirichletProcess):

    def __init__(self, name, Xs, *args, **kwargs):
        pass


if __name__ == "__main__":

    Xs = np.array([-1, 0, 1])
    Xnew = np.array([-3, -1, 0.5, 3.2])
    K = 20

    with pm.Model() as model:
        try:
            Xs.shape[1]
        except IndexError as e:
            Xs = Xs[..., np.newaxis]

        try:
            Xnew.shape[1]
        except IndexError as e:
            Xnew = Xnew[..., np.newaxis]

        base_dist = pm.Normal("G0", 0, 1, shape=(1, K))
        dp = pm.dp.DirichletProcess(
            name="dp",
            alpha=1,
            base_dist=base_dist,
            K=K,
        )
        # dp.prior("dp-post", Xs)
        dp.posterior("dp-post", Xs, Xnew)

        trace = pm.sample(
            draws=1000,
            chains=1,
        )

        print(trace.to_dict()["posterior"]["dp-post"][0].mean(axis=0))
