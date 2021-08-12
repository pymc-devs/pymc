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
import pandas as pd

from aesara import tensor as at

import pymc3 as pm

import aesara
from aesara.tensor.random.op import RandomVariable

aesara.config.exception_verbosity = "high"

from pymc3.math import logsumexp

import argparse

__all__ = [
    "DirichletProcess",
]

parser = argparse.ArgumentParser()
parser.add_argument("--dp", action="store_true")
parser.add_argument("--dpmix", action="store_true")
args = parser.parse_args()


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

    return betas * sticks


def mixture_logp(weights, comp_dist, atoms, obs):
    # comp_dist is something like lambda obs, atoms: pm.Normal.logp(obs, atoms, 1)

    atoms_shape = atoms.shape.eval()
    obs_shape = obs.shape.eval()

    # assert atoms.shape == weights.shape

    weights = at.broadcast_to(weights, shape=obs_shape + atoms_shape)
    log_weights = (at.log(weights)).T
    atoms_logp = comp_dist(
        at.broadcast_to(obs, atoms_shape + obs_shape),
        at.broadcast_to(atoms, obs_shape + atoms_shape).T,
    )

    return logsumexp(log_weights + atoms_logp)


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
        elif isinstance(alpha, (np.ndarray, aesara.graph.basic.Variable)):
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

        betas = pm.Beta("betas", 1., self.alpha, shape=(self.K,))
        self.weights = pm.Deterministic("weights", stick_breaking(betas))

    def __add__(self, other):
        return self.__class__(self.alpha, self.base_dist)

    def __str__(self):
        return self.name

    def prior(self, name, Xs, **kwargs):
        G0 = self._build_prior(name, Xs, **kwargs)

        return G0

    def _build_prior(self, name, Xs, *args, **kwargs):
        """
        Estimates CDF for specific Xs
        """
        self.Xs = Xs



        prior_atoms = pm.Deterministic(
            name=name,
            var=linear_comb,
        )

        return prior_atoms

    def prior(self, name, Xs, **kwargs):
        # insert some checks for errors

        G0 = self._build_prior(name, Xs, **kwargs)

        return G0

    def _build_posterior(self, name, Xs, Xnew, *args, **kwargs):

        Xs = at.atleast_2d(Xs).T
        Xnew = at.atleast_2d(Xnew).T

        N = Xs.shape.eval()[0]

        dirac = at.sum(at.ge(Xnew, Xs.T), axis=1)  # shape = (N',)
        dirac = at.as_tensor_variable(dirac)  # shape = (N',)

        base_dist = pm.Normal("G0", 0, 3, shape=(K, 1))  # K draws
        weights = pm.Dirichlet(
            name="sticks",
            a=np.ones(shape=(K,)),
        )

        empirical_base_cdf = at.le(base_dist, Xnew.T)
        empirical_base_cdf = at.sum(at.mul(empirical_base_cdf.T, weights), axis=1)

        posterior_atoms = pm.Deterministic(
            name="posterior-dp",
            var=empirical_base_cdf / (1 + N) + dirac / (1 + N),
        )

        return posterior_atoms

    def posterior(self, name, Xs, Xnew, **kwargs):

        Gn = self._build_posterior(name, Xs, Xnew, **kwargs)

        return Gn


class DirichletProcessMixture(DirichletProcess):

    def __init__(self, name, alpha, atoms, base_dist_class, K, **kwargs):
        # base_dist_class is temporary
        # should atoms be a dict?
        super().__init__(name, alpha, base_dist_class("base-dist", atoms), K)

        # what if atom is not 1-D
        kernel = lambda obs, atom: base_dist_class.logp(obs, atom)

        logp = lambda x: mixture_logp(
            weights=self.weights,
            comp_dist=kernel,
            atoms=self.base_dist,
            observed=x,
        )

        self.mixture = pm.Potential(
            name="mixture",
            var=logp,
            **kwargs,
        )


if __name__ == "__main__":

    Xs = np.array([-1, 0, 1])
    Xnew = np.array([-3, -1, 0.5, 3.2, 4])
    K = 19

    sunspot_df = pd.read_csv(
        pm.get_data("sunspot.csv"), sep=";", names=["time", "sunspot.year"], usecols=[0, 1]
    )

    if args.dp:
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

    if args.dpmix:
        # not ready yet
        with pm.Model() as dp_mixture_model:
            base_dist = pm.Normal("G0", 0, 3, shape=(K,))
            dp_mix = DirichletProcessMixture(
                name="dp-mix",
                alpha=1,
                atoms=pm.Gamma("lambda_", 300.0, 2.0, shape=K),
                base_dist_class=pm.Poisson,
                K=K,
                observed=sunspot_df["sunspot.year"],
            )
            pm.sample(1000)
            print("GREAT CODING")
