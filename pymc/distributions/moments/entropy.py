#   Copyright 2024 - present The PyMC Developers
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

"""Entropy dispatcher for pymc random variables.

The entropy of a random variable is the differential entropy for continuous
variables and the Shannon entropy (in nats) for discrete variables, matching
the convention of :meth:`scipy.stats.rv_continuous.entropy`.
"""

from functools import singledispatch

import numpy as np

from pytensor import tensor as pt
from pytensor.tensor.random.basic import (
    BernoulliRV,
    BetaRV,
    CategoricalRV,
    CauchyRV,
    DirichletRV,
    ExponentialRV,
    GammaRV,
    GeometricRV,
    GumbelRV,
    HalfNormalRV,
    InvGammaRV,
    LaplaceRV,
    LogisticRV,
    LogNormalRV,
    MvNormalRV,
    NormalRV,
    ParetoRV,
    StudentTRV,
    TriangularRV,
    UniformRV,
    VonMisesRV,
)
from pytensor.tensor.variable import TensorVariable

from pymc.distributions.continuous import (
    AsymmetricLaplaceRV,
    HalfCauchyRV,
    MoyalRV,
    WeibullBetaRV,
)
from pymc.distributions.discrete import DiscreteUniformRV
from pymc.distributions.shape_utils import maybe_resize, rv_size_is_none

__all__ = ["entropy"]


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


@singledispatch
def _entropy(op, rv, *rv_inputs) -> TensorVariable:
    raise NotImplementedError(f"Variable {rv} of type {op} has no entropy implementation.")


def entropy(rv: TensorVariable) -> TensorVariable:
    """Compute the entropy of a random variable.

    The entropy is the differential entropy for continuous distributions and
    the Shannon entropy (in nats) for discrete distributions. This matches the
    convention used by ``scipy.stats``'s ``entropy`` method.

    The only parameter to this function is the RandomVariable
    for which the entropy is to be derived.
    """
    return _entropy(rv.owner.op, rv, *rv.owner.inputs)


# --- Continuous univariate ---


@_entropy.register(NormalRV)
def normal_entropy(op, rv, rng, size, mu, sigma):
    return maybe_resize(0.5 * pt.log(2 * np.pi * np.e) + pt.log(sigma), size)


@_entropy.register(UniformRV)
def uniform_entropy(op, rv, rng, size, lower, upper):
    return maybe_resize(pt.log(upper - lower), size)


@_entropy.register(ExponentialRV)
def exponential_entropy(op, rv, rng, size, mu):
    # ``mu`` is the mean (scale) of the distribution
    return maybe_resize(1 + pt.log(mu), size)


@_entropy.register(LaplaceRV)
def laplace_entropy(op, rv, rng, size, mu, b):
    return maybe_resize(1 + pt.log(2 * b), size)


@_entropy.register(AsymmetricLaplaceRV)
def asymmetric_laplace_entropy(op, rv, rng, size, b, kappa, mu):
    # scale = 1 / b
    return maybe_resize(1 + pt.log((kappa + 1 / kappa) / b), size)


@_entropy.register(CauchyRV)
def cauchy_entropy(op, rv, rng, size, alpha, beta):
    return maybe_resize(pt.log(4 * np.pi * beta), size)


@_entropy.register(HalfCauchyRV)
def halfcauchy_entropy(op, rv, rng, size, beta):
    return maybe_resize(pt.log(2 * np.pi * beta), size)


@_entropy.register(HalfNormalRV)
def halfnormal_entropy(op, rv, rng, size, loc, sigma):
    return maybe_resize(0.5 * pt.log(np.pi * sigma**2 / 2) + 0.5, size)


@_entropy.register(GammaRV)
def gamma_entropy(op, rv, rng, size, alpha, inv_beta):
    # ``inv_beta`` is the scale (1 / rate)
    return maybe_resize(
        alpha + pt.log(inv_beta) + pt.gammaln(alpha) + (1 - alpha) * pt.digamma(alpha),
        size,
    )


@_entropy.register(InvGammaRV)
def invgamma_entropy(op, rv, rng, size, alpha, beta):
    # ``beta`` is the scale
    return maybe_resize(
        alpha + pt.log(beta) + pt.gammaln(alpha) - (1 + alpha) * pt.digamma(alpha),
        size,
    )


@_entropy.register(BetaRV)
def beta_entropy(op, rv, rng, size, alpha, beta):
    return maybe_resize(
        _betaln(alpha, beta)
        - (alpha - 1) * pt.digamma(alpha)
        - (beta - 1) * pt.digamma(beta)
        + (alpha + beta - 2) * pt.digamma(alpha + beta),
        size,
    )


@_entropy.register(LogisticRV)
def logistic_entropy(op, rv, rng, size, mu, s):
    return maybe_resize(pt.log(s) + 2, size)


@_entropy.register(LogNormalRV)
def lognormal_entropy(op, rv, rng, size, mu, sigma):
    return maybe_resize(mu + 0.5 * pt.log(2 * np.pi * np.e * sigma**2), size)


@_entropy.register(GumbelRV)
def gumbel_entropy(op, rv, rng, size, mu, beta):
    return maybe_resize(pt.log(beta) + np.euler_gamma + 1, size)


@_entropy.register(ParetoRV)
def pareto_entropy(op, rv, rng, size, alpha, m):
    return maybe_resize(pt.log(m / alpha) + 1 / alpha + 1, size)


@_entropy.register(WeibullBetaRV)
def weibull_entropy(op, rv, rng, size, alpha, beta):
    # ``alpha`` is the shape, ``beta`` is the scale
    return maybe_resize(np.euler_gamma * (1 - 1 / alpha) + pt.log(beta / alpha) + 1, size)


@_entropy.register(StudentTRV)
def studentt_entropy(op, rv, rng, size, nu, mu, sigma):
    return maybe_resize(
        pt.log(sigma)
        + 0.5 * (nu + 1) * (pt.digamma((nu + 1) / 2) - pt.digamma(nu / 2))
        + pt.log(pt.sqrt(nu))
        + _betaln(nu / 2, 0.5),
        size,
    )


@_entropy.register(TriangularRV)
def triangular_entropy(op, rv, rng, size, lower, c, upper):
    return maybe_resize(0.5 + pt.log((upper - lower) / 2), size)


@_entropy.register(MoyalRV)
def moyal_entropy(op, rv, rng, size, mu, sigma):
    return maybe_resize(
        pt.log(sigma) + 0.5 * pt.log(2 * np.pi) + 0.5 * (np.euler_gamma + np.log(2) + 1),
        size,
    )


@_entropy.register(VonMisesRV)
def vonmises_entropy(op, rv, rng, size, mu, kappa):
    return maybe_resize(
        pt.log(2 * np.pi * pt.i0(kappa)) - kappa * pt.i1(kappa) / pt.i0(kappa), size
    )


# --- Discrete univariate ---


@_entropy.register(BernoulliRV)
def bernoulli_entropy(op, rv, rng, size, p):
    return maybe_resize(-p * pt.log(p) - (1 - p) * pt.log1p(-p), size)


@_entropy.register(GeometricRV)
def geometric_entropy(op, rv, rng, size, p):
    return maybe_resize((-(1 - p) * pt.log1p(-p) - p * pt.log(p)) / p, size)


@_entropy.register(DiscreteUniformRV)
def discrete_uniform_entropy(op, rv, rng, size, lower, upper):
    return maybe_resize(pt.log(upper - lower + 1), size)


@_entropy.register(CategoricalRV)
def categorical_entropy(op, rv, rng, size, p):
    return maybe_resize(-pt.sum(p * pt.log(p), axis=-1), size)


# --- Multivariate ---


@_entropy.register(MvNormalRV)
def mvnormal_entropy(op, rv, rng, size, mu, cov):
    k = cov.shape[-1]
    _, logdet = pt.linalg.slogdet(cov)
    res = 0.5 * k * pt.log(2 * np.pi * np.e) + 0.5 * logdet
    if rv_size_is_none(size):
        return res
    return maybe_resize(res, size)


@_entropy.register(DirichletRV)
def dirichlet_entropy(op, rv, rng, size, a):
    a0 = pt.sum(a, axis=-1)
    k = a.shape[-1]
    log_beta = pt.sum(pt.gammaln(a), axis=-1) - pt.gammaln(a0)
    res = log_beta + (a0 - k) * pt.digamma(a0) - pt.sum((a - 1) * pt.digamma(a), axis=-1)
    if rv_size_is_none(size):
        return res
    return maybe_resize(res, size)
