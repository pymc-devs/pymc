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

"""Mean dispatcher for pymc random variables."""

from functools import singledispatch

import numpy as np

from pytensor import tensor as pt
from pytensor.tensor.math import tanh
from pytensor.tensor.random.basic import (
    BernoulliRV,
    BetaBinomialRV,
    BetaRV,
    BinomialRV,
    CategoricalRV,
    CauchyRV,
    DirichletRV,
    ExponentialRV,
    GammaRV,
    GeometricRV,
    GumbelRV,
    HalfCauchyRV,
    HalfNormalRV,
    HyperGeometricRV,
    InvGammaRV,
    LaplaceRV,
    LogisticRV,
    LogNormalRV,
    MultinomialRV,
    MvNormalRV,
    NegBinomialRV,
    NormalRV,
    ParetoRV,
    PoissonRV,
    StudentTRV,
    TriangularRV,
    UniformRV,
    VonMisesRV,
)
from pytensor.tensor.variable import TensorVariable

from pymc.distributions.continuous import (
    AsymmetricLaplaceRV,
    ExGaussianRV,
    FlatRV,
    HalfFlatRV,
    HalfStudentTRV,
    KumaraswamyRV,
    MoyalRV,
    PolyaGammaRV,
    RiceRV,
    SkewNormalRV,
    SkewStudentTRV,
    WaldRV,
    WeibullBetaRV,
)
from pymc.distributions.discrete import DiscreteUniformRV
from pymc.distributions.distribution import DiracDeltaRV
from pymc.distributions.mixture import MixtureRV
from pymc.distributions.multivariate import (
    CARRV,
    DirichletMultinomialRV,
    KroneckerNormalRV,
    LKJCorrRV,
    MatrixNormalRV,
    MvStudentTRV,
    StickBreakingWeightsRV,
    _LKJCholeskyCovRV,
)
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.exceptions import UndefinedMomentException

__all__ = ["mean"]


@singledispatch
def _mean(op, rv, *rv_inputs) -> TensorVariable:
    raise NotImplementedError(f"Variable {rv} of type {op} has no mean implementation.")


def mean(rv: TensorVariable) -> TensorVariable:
    """Compute the expected value of a random variable.

    The only parameter to this function is the RandomVariable
    for which the value is to be derived.
    """
    return _mean(rv.owner.op, rv, *rv.owner.inputs)


def maybe_resize(a: TensorVariable, size) -> TensorVariable:
    if not rv_size_is_none(size):
        a = pt.full(size, a)
    return a


@_mean.register(AsymmetricLaplaceRV)
def asymmetric_laplace_mean(op, rv, rng, size, b, kappa, mu):
    return maybe_resize(mu - (kappa - 1 / kappa) / b, size)


@_mean.register(BernoulliRV)
def bernoulli_mean(op, rv, rng, size, p):
    return maybe_resize(p, size)


@_mean.register(BetaRV)
def beta_mean(op, rv, rng, size, alpha, beta):
    return maybe_resize(alpha / (alpha + beta), size)


@_mean.register(BetaBinomialRV)
def betabinomial_mean(op, rv, rng, size, n, alpha, beta):
    return maybe_resize((n * alpha) / (alpha + beta), size)


@_mean.register(BinomialRV)
def binomial_mean(op, rv, rng, size, n, p):
    return maybe_resize(n * p, size)


@_mean.register(CARRV)
def car_mean(op, rv, rng, size, mu, W, alpha, tau, W_is_valid):
    return pt.full_like(rv, mu)


@_mean.register(CategoricalRV)
def categorical_mean(op, rv, *args):
    raise UndefinedMomentException("The mean of the Categorical distribution is undefined")


@_mean.register(CauchyRV)
def cauchy_mean(op, rv, rng, size, alpha, beta):
    raise UndefinedMomentException("The mean of the Cauchy distribution is undefined")


@_mean.register(DiracDeltaRV)
def dirac_delta_mean(op, rv, size, c):
    return maybe_resize(c, size)


@_mean.register(DirichletRV)
def dirichlet_mean(op, rv, rng, size, a):
    norm_constant = pt.sum(a, axis=-1)[..., None]
    mean = a / norm_constant
    if not rv_size_is_none(size):
        mean = pt.full(pt.concatenate([size, [a.shape[-1]]]), mean)
    return mean


@_mean.register(DirichletMultinomialRV)
def dirichlet_multinomial_mean(op, rv, rng, size, n, a):
    mean = pt.shape_padright(n) * a / pt.sum(a, axis=-1, keepdims=True)
    if not rv_size_is_none(size):
        output_size = pt.concatenate([size, [a.shape[-1]]])
        # We can't use pt.full because output_size is symbolic
        mean, _ = pt.broadcast_arrays(mean, pt.zeros(size)[..., None])
    return mean


@_mean.register(DiscreteUniformRV)
def discrete_uniform_mean(op, rv, rng, size, lower, upper):
    return maybe_resize((upper + lower) / 2.0, size)


@_mean.register(ExGaussianRV)
def exgaussian_mean(op, rv, rng, size, mu, nu, sigma):
    mu, nu, _ = pt.broadcast_arrays(mu, nu, sigma)
    return maybe_resize(mu + nu, size)


@_mean.register(ExponentialRV)
def exponential_mean(op, rv, rng, size, mu):
    return maybe_resize(mu, size)


@_mean.register(FlatRV)
def flat_mean(op, rv, *args):
    raise UndefinedMomentException("The mean of the Flat distribution is undefined")


@_mean.register(GammaRV)
def gamma_mean(op, rv, rng, size, alpha, inv_beta):
    # The pytensor `GammaRV` `Op` inverts the `beta` parameter itself
    return maybe_resize(alpha * inv_beta, size)


@_mean.register(GeometricRV)
def geometric_mean(op, rv, rng, size, p):
    return maybe_resize(1.0 / p, size)


@_mean.register(GumbelRV)
def gumbel_mean(op, rv, rng, size, mu, beta):
    return maybe_resize(mu + beta * np.euler_gamma, size)


@_mean.register(HalfStudentTRV)
def half_studentt_mean(op, rv, rng, size, nu, sigma):
    return maybe_resize(
        pt.switch(
            nu > 1,
            2
            * sigma
            * pt.sqrt(nu / np.pi)
            * pt.exp(pt.gammaln(0.5 * (nu + 1)) - pt.gammaln(nu / 2) - pt.log(nu - 1)),
            np.nan,
        ),
        size,
    )


@_mean.register(HalfCauchyRV)
def halfcauchy_mean(op, rv, rng, size, loc, beta):
    raise UndefinedMomentException("The mean of the HalfCauchy distribution is undefined")


@_mean.register(HalfFlatRV)
def halfflat_mean(op, rv, *args):
    raise UndefinedMomentException("The mean of the HalfFlat distribution is undefined")


@_mean.register(HalfNormalRV)
def halfnormal_mean(op, rv, rng, size, loc, sigma):
    _, sigma = pt.broadcast_arrays(loc, sigma)
    return maybe_resize(sigma * pt.sqrt(2 / np.pi), size)


@_mean.register(HyperGeometricRV)
def hypergeometric_mean(op, rv, rng, size, good, bad, n):
    N, k = good + bad, good
    return maybe_resize(n * k / N, size)


@_mean.register(InvGammaRV)
def invgamma_mean(op, rv, rng, size, alpha, beta):
    return maybe_resize(pt.switch(alpha > 1, beta / (alpha - 1.0), np.nan), size)


@_mean.register(KroneckerNormalRV)
def kronecker_normal_mean(op, rv, rng, size, mu, covs, chols, evds):
    mean = mu
    if not rv_size_is_none(size):
        mean_size = pt.concatenate([size, mu.shape])
        mean = pt.full(mean_size, mu)
    return mean


@_mean.register(KumaraswamyRV)
def kumaraswamy_mean(op, rv, rng, size, a, b):
    return maybe_resize(
        pt.exp(pt.log(b) + pt.gammaln(1 + 1 / a) + pt.gammaln(b) - pt.gammaln(1 + 1 / a + b)),
        size,
    )


@_mean.register(LaplaceRV)
def laplace_mean(op, rv, rng, size, mu, b):
    return maybe_resize(pt.broadcast_arrays(mu, b)[0], size)


@_mean.register(_LKJCholeskyCovRV)
def lkj_cholesky_cov_mean(op, rv, rng, n, eta, sd_dist):
    diag_idxs = (pt.cumsum(pt.arange(1, n + 1)) - 1).astype("int32")
    mean = pt.zeros_like(rv)
    mean = pt.set_subtensor(mean[..., diag_idxs], 1)
    return mean


@_mean.register(LKJCorrRV)
def lkj_corr_mean(op, rv, rng, size, *args):
    return pt.full_like(rv, pt.eye(rv.shape[-1]))


@_mean.register(LogisticRV)
def logistic_mean(op, rv, rng, size, mu, s):
    return maybe_resize(pt.broadcast_arrays(mu, s)[0], size)


@_mean.register(LogNormalRV)
def lognormal_mean(op, rv, rng, size, mu, sigma):
    return maybe_resize(pt.exp(mu + 0.5 * sigma**2), size)


@_mean.register(MixtureRV)
def marginal_mixture_mean(op, rv, rng, weights, *components):
    ndim_supp = components[0].owner.op.ndim_supp
    weights = pt.shape_padright(weights, ndim_supp)
    mix_axis = -ndim_supp - 1

    if len(components) == 1:
        mean_components = mean(components[0])

    else:
        mean_components = pt.stack(
            [mean(component) for component in components],
            axis=mix_axis,
        )

    return pt.sum(weights * mean_components, axis=mix_axis)


@_mean.register(MatrixNormalRV)
def matrix_normal_mean(op, rv, rng, size, mu, rowchol, colchol):
    return pt.full_like(rv, mu)


@_mean.register(MoyalRV)
def moyal_mean(op, rv, rng, size, mu, sigma):
    return maybe_resize(mu + sigma * (np.euler_gamma + pt.log(2)), size)


@_mean.register(MultinomialRV)
def multinomial_mean(op, rv, rng, size, n, p):
    n = pt.shape_padright(n)
    mean = n * p
    if not rv_size_is_none(size):
        output_size = pt.concatenate([size, [p.shape[-1]]])
        mean = pt.full(output_size, mean)
    return mean


@_mean.register(MvNormalRV)
def mvnormal_mean(op, rv, rng, size, mu, cov):
    mean = mu
    if not rv_size_is_none(size):
        mean_size = pt.concatenate([size, [mu.shape[-1]]])
        mean = pt.full(mean_size, mu)
    return mean


@_mean.register(MvStudentTRV)
def mvstudentt_mean(op, rv, rng, size, nu, mu, scale):
    mean = mu
    if not rv_size_is_none(size):
        mean_size = pt.concatenate([size, [mu.shape[-1]]])
        mean = pt.full(mean_size, mean)
    return mean


@_mean.register(NegBinomialRV)
def negative_binomial_mean(op, rv, rng, size, n, p):
    return maybe_resize(n * (1 - p) / p, size)


@_mean.register(NormalRV)
def normal_mean(op, rv, rng, size, mu, sigma):
    return maybe_resize(pt.broadcast_arrays(mu, sigma)[0], size)


@_mean.register(ParetoRV)
def pareto_mean(op, rv, rng, size, alpha, m):
    return maybe_resize(pt.switch(alpha > 1, alpha * m / (alpha - 1), np.nan), size)


@_mean.register(PoissonRV)
def poisson_mean(op, rv, rng, size, mu):
    return maybe_resize(mu, size)


@_mean.register(PolyaGammaRV)
def polya_gamma_mean(op, rv, rng, size, h, z):
    return maybe_resize(pt.switch(pt.eq(z, 0), h / 4, tanh(z / 2) * (h / (2 * z))), size)


@_mean.register(RiceRV)
def rice_mean(op, rv, rng, size, nu, sigma):
    nu_sigma_ratio = -(nu**2) / (2 * sigma**2)
    return maybe_resize(
        sigma
        * np.sqrt(np.pi / 2)
        * pt.exp(nu_sigma_ratio / 2)
        * (
            (1 - nu_sigma_ratio) * pt.i0(-nu_sigma_ratio / 2)
            - nu_sigma_ratio * pt.i1(-nu_sigma_ratio / 2)
        ),
        size,
    )


@_mean.register(SkewNormalRV)
def skew_normal_mean(op, rv, rng, size, mu, sigma, alpha):
    return maybe_resize(mu + sigma * (2 / np.pi) ** 0.5 * alpha / (1 + alpha**2) ** 0.5, size)


@_mean.register(SkewStudentTRV)
def skew_studentt_mean(op, rv, rng, size, a, b, mu, sigma):
    a, b, mu, _ = pt.broadcast_arrays(a, b, mu, sigma)
    Et = mu + (a - b) * pt.sqrt(a + b) * pt.gamma(a - 0.5) * pt.gamma(b - 0.5) / (
        2 * pt.gamma(a) * pt.gamma(b)
    )
    if not rv_size_is_none(size):
        Et = pt.full(size, Et)
    return Et


@_mean.register(StickBreakingWeightsRV)
def stick_breaking_mean(op, rv, rng, size, alpha, K):
    K = K.squeeze()
    alpha = alpha[..., np.newaxis]
    mean = (alpha / (1 + alpha)) ** pt.arange(K)
    mean *= 1 / (1 + alpha)
    mean = pt.concatenate([mean, (alpha / (1 + alpha)) ** K], axis=-1)
    if not rv_size_is_none(size):
        mean_size = pt.concatenate(
            [
                size,
                [
                    K + 1,
                ],
            ]
        )
        mean = pt.full(mean_size, mean)
    return mean


@_mean.register(StudentTRV)
def studentt_mean(op, rv, rng, size, nu, mu, sigma):
    return maybe_resize(pt.broadcast_arrays(mu, nu, sigma)[0], size)


@_mean.register(TriangularRV)
def triangular_mean(op, rv, rng, size, lower, c, upper):
    return maybe_resize((lower + upper + c) / 3, size)


@_mean.register(UniformRV)
def uniform_mean(op, rv, rng, size, lower, upper):
    return maybe_resize((lower + upper) / 2, size)


@_mean.register(VonMisesRV)
def vonmisses_mean(op, rv, rng, size, mu, kappa):
    return maybe_resize(pt.broadcast_arrays(mu, kappa)[0], size)


@_mean.register(WaldRV)
def wald_mean(op, rv, rng, size, mu, lam, alpha):
    return maybe_resize(pt.broadcast_arrays(mu, lam, alpha)[0], size)


@_mean.register(WeibullBetaRV)
def weibull_mean(op, rv, rng, size, alpha, beta):
    return maybe_resize(beta * pt.gamma(1 + 1 / alpha), size)
