#   Copyright 2026 - present The PyMC Developers
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
from functools import singledispatch

import numpy as np

from pytensor import tensor as pt
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
    HalfCauchyRV,
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
from pymc.distributions.moments.means import mean
from pymc.distributions.multivariate import (
    CARRV,
    DirichletMultinomialRV,
    KroneckerNormalRV,
    LKJCorrRV,
    MatrixNormalRV,
    MvStudentTRV,
    StickBreakingWeightsRV,
    WishartRV,
    _LKJCholeskyCovRV,
)
from pymc.distributions.shape_utils import maybe_resize, rv_size_is_none
from pymc.exceptions import UndefinedMomentException

__all__ = ["variance"]


@singledispatch
def _variance(op, rv, *rv_inputs) -> TensorVariable:
    raise NotImplementedError(f"Variable {rv} of type {op} has no variance implementation.")


def variance(rv: TensorVariable) -> TensorVariable:
    """Compute the variance of a random variable.

    The only parameter to this function is the RandomVariable
    for which the value is to be derived.
    """
    return _variance(rv.owner.op, rv, *rv.owner.inputs)


@_variance.register(AsymmetricLaplaceRV)
def asymmetric_laplace_variance(op, rv, rng, size, b, kappa, mu):
    return maybe_resize((1 + kappa**4) / (b**2 * kappa**2), size)


@_variance.register(BernoulliRV)
def bernoulli_variance(op, rv, rng, size, p):
    return maybe_resize(p * (1 - p), size)


@_variance.register(BetaRV)
def beta_variance(op, rv, rng, size, alpha, beta):
    alpha, beta = pt.cast(alpha, "floatX"), pt.cast(beta, "floatX")
    return maybe_resize((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1)), size)


@_variance.register(BetaBinomialRV)
def betabinomial_variance(op, rv, rng, size, n, alpha, beta):
    n, alpha, beta = pt.cast(n, "floatX"), pt.cast(alpha, "floatX"), pt.cast(beta, "floatX")
    return maybe_resize(
        (n * alpha * beta * (alpha + beta + n)) / ((alpha + beta) ** 2 * (alpha + beta + 1)),
        size,
    )


@_variance.register(BinomialRV)
def binomial_variance(op, rv, rng, size, n, p):
    return maybe_resize(n * p * (1 - p), size)


@_variance.register(CARRV)
def car_variance(op, rv, rng, size, mu, W, alpha, tau, W_is_valid):
    W = pt.as_tensor_variable(W)
    rho = pt.as_tensor_variable(alpha)
    tau = pt.as_tensor_variable(tau)

    D = pt.diag(pt.sum(W, axis=-1))
    Q = tau * (D - rho * W)

    return pt.linalg.inv(Q)


@_variance.register(CategoricalRV)
def categorical_variance(op, rv, *args):
    raise UndefinedMomentException("The variance of the Categorical distribution is undefined")


@_variance.register(CauchyRV)
def cauchy_variance(op, rv, rng, size, alpha, beta):
    raise UndefinedMomentException("The variance of the Cauchy distribution is undefined")


@_variance.register(DiracDeltaRV)
def dirac_delta_variance(op, rv, size, c):
    return maybe_resize(pt.zeros_like(c), size)


@_variance.register(DirichletRV)
def dirichlet_variance(op, rv, rng, size, a):
    a0 = pt.sum(a, axis=-1, keepdims=True)
    var = (a * (a0 - a)) / (a0**2 * (a0 + 1))
    if not rv_size_is_none(size):
        var = pt.full(pt.concatenate([size, [a.shape[-1]]]), var)
    return var


@_variance.register(DirichletMultinomialRV)
def dirichlet_multinomial_variance(op, rv, rng, size, n, a):
    a0 = pt.sum(a, axis=-1, keepdims=True)
    n_expanded = pt.shape_padright(n)
    var = n_expanded * (a / a0) * (1 - a / a0) * (a0 + n_expanded) / (a0 + 1)
    if not rv_size_is_none(size):
        var, _ = pt.broadcast_arrays(var, pt.zeros(size)[..., None])
    return var


@_variance.register(DiscreteUniformRV)
def discrete_uniform_variance(op, rv, rng, size, lower, upper):
    return maybe_resize(((upper - lower + 1) ** 2 - 1) / 12.0, size)


@_variance.register(ExGaussianRV)
def exgaussian_variance(op, rv, rng, size, mu, nu, sigma):
    _, nu, sigma = pt.broadcast_arrays(mu, nu, sigma)
    return maybe_resize(sigma**2 + nu**2, size)


@_variance.register(ExponentialRV)
def exponential_variance(op, rv, rng, size, mu):
    return maybe_resize(mu**2, size)


@_variance.register(FlatRV)
def flat_variance(op, rv, rng, size):
    raise UndefinedMomentException("The variance of Flat distribution is undefined")


@_variance.register(GammaRV)
def gamma_variance(op, rv, rng, size, alpha, inv_beta):
    # The pytensor `GammaRV` `Op` inverts the `beta` parameter itself
    return maybe_resize(alpha * inv_beta**2, size)


@_variance.register(GeometricRV)
def geometric_variance(op, rv, rng, size, p):
    return maybe_resize((1 - p) / p**2, size)


@_variance.register(GumbelRV)
def gumbel_variance(op, rv, rng, size, mu, beta):
    return maybe_resize(np.pi**2 / 6 * beta**2, size)


@_variance.register(HalfStudentTRV)
def half_studentt_variance(op, rv, rng, size, nu, sigma):
    mean_val = (
        2
        * sigma
        * pt.sqrt(nu / np.pi)
        * pt.exp(pt.gammaln(0.5 * (nu + 1)) - pt.gammaln(nu / 2) - pt.log(nu - 1))
    )
    return maybe_resize(
        pt.switch(
            nu > 2,
            sigma**2 * nu / (nu - 2) - mean_val**2,
            np.nan,
        ),
        size,
    )


@_variance.register(HalfCauchyRV)
def pymc_halfcauchy_variance(op, rv, rng, size, beta):
    raise UndefinedMomentException("The variance of the HalfCauchy distribution is undefined")


@_variance.register(HalfFlatRV)
def halfflat_variance(op, rv, rng, size):
    raise UndefinedMomentException("The variance of the HalfFlat distribution is undefined")


@_variance.register(HalfNormalRV)
def halfnormal_variance(op, rv, rng, size, loc, sigma):
    _, sigma = pt.broadcast_arrays(loc, sigma)
    return maybe_resize(sigma**2 * (1 - 2 / np.pi), size)


@_variance.register(HyperGeometricRV)
def hypergeometric_variance(op, rv, rng, size, good, bad, n):
    N = good + bad
    K = good
    return maybe_resize(n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1)), size)


@_variance.register(InvGammaRV)
def invgamma_variance(op, rv, rng, size, alpha, beta):
    return maybe_resize(
        pt.switch(alpha > 2, beta**2 / ((alpha - 1) ** 2 * (alpha - 2)), np.nan), size
    )


@_variance.register(KroneckerNormalRV)
def kronecker_normal_variance(op, rv, rng, size, mu, sigma, *covs):
    from functools import reduce

    cov = reduce(pt.linalg.kron, covs)
    cov = cov + sigma**2 * pt.eye(cov.shape[-2])
    if not rv_size_is_none(size):
        cov_size = pt.concatenate([size, cov.shape[-2:]])
        cov = pt.full(cov_size, cov)
    return cov


@_variance.register(KumaraswamyRV)
def kumaraswamy_variance(op, rv, rng, size, a, b):
    m1 = pt.exp(pt.log(b) + pt.gammaln(1 + 1 / a) + pt.gammaln(b) - pt.gammaln(1 + 1 / a + b))
    m2 = pt.exp(pt.log(b) + pt.gammaln(1 + 2 / a) + pt.gammaln(b) - pt.gammaln(1 + 2 / a + b))
    return maybe_resize(m2 - m1**2, size)


@_variance.register(LaplaceRV)
def laplace_variance(op, rv, rng, size, mu, b):
    return maybe_resize(2 * b**2, size)


@_variance.register(_LKJCholeskyCovRV)
def lkj_cholesky_cov_variance(op, rv, rng, n, eta, sd_dist):
    raise NotImplementedError(
        "LKJCholeskyCov variance not implemented due to complex covariance structure."
    )


@_variance.register(LKJCorrRV)
def lkj_corr_variance(op, rv, rng, size, *args):
    raise NotImplementedError(
        "LKJCorr variance not implemented due to complex covariance structure."
    )


@_variance.register(LogisticRV)
def logistic_variance(op, rv, rng, size, mu, s):
    return maybe_resize(np.pi**2 / 3 * s**2, size)


@_variance.register(LogNormalRV)
def lognormal_variance(op, rv, rng, size, mu, sigma):
    return maybe_resize((pt.exp(sigma**2) - 1) * pt.exp(2 * mu + sigma**2), size)


@_variance.register(MixtureRV)
def marginal_mixture_variance(op, rv, rng, weights, *components):
    ndim_supp = components[0].owner.op.ndim_supp
    weights = pt.shape_padright(weights, ndim_supp)
    mix_axis = -ndim_supp - 1

    if len(components) == 1:
        mean_components = mean(components[0])
        var_components = variance(components[0])
    else:
        mean_components = pt.stack(
            [mean(component) for component in components],
            axis=mix_axis,
        )
        var_components = pt.stack(
            [variance(component) for component in components],
            axis=mix_axis,
        )

    # Law of total variance: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
    mixture_mean = pt.sum(weights * mean_components, axis=mix_axis)
    mean_of_variance = pt.sum(weights * var_components, axis=mix_axis)
    variance_of_mean = pt.sum(
        weights * (mean_components - pt.shape_padright(mixture_mean, 1)) ** 2, axis=mix_axis
    )
    return mean_of_variance + variance_of_mean


@_variance.register(MatrixNormalRV)
def matrix_normal_variance(op, rv, rng, size, mu, rowchol, colchol):
    raise NotImplementedError(
        "MatrixNormal variance not implemented due to complex covariance structure."
    )


@_variance.register(MoyalRV)
def moyal_variance(op, rv, rng, size, mu, sigma):
    return maybe_resize(np.pi**2 / 2 * sigma**2, size)


@_variance.register(MultinomialRV)
def multinomial_variance(op, rv, rng, size, n, p):
    # Covariance matrix: Cov(X_i, X_j) = n * (delta_ij * p_i - p_i * p_j)
    k = p.shape[-1]
    diag_p = p[..., :, None] * pt.eye(k, dtype=p.dtype)
    outer_pp = p[..., :, None] * p[..., None, :]
    cov = n[..., None, None] * (diag_p - outer_pp)

    if not rv_size_is_none(size):
        output_size = pt.concatenate([size, cov.shape[-2:]])
        cov = pt.full(output_size, cov)
    return cov


@_variance.register(MvNormalRV)
def mvnormal_variance(op, rv, rng, size, mu, cov):
    if not rv_size_is_none(size):
        cov_size = pt.concatenate([size, cov.shape[-2:]])
        cov = pt.full(cov_size, cov)
    return cov


@_variance.register(MvStudentTRV)
def mvstudentt_variance(op, rv, rng, size, nu, mu, scale):
    cov = pt.switch(nu > 2, (nu / (nu - 2)) * scale, np.nan)
    if not rv_size_is_none(size):
        cov_size = pt.concatenate([size, scale.shape[-2:]])
        cov = pt.full(cov_size, cov)
    return cov


@_variance.register(NegBinomialRV)
def negative_binomial_variance(op, rv, rng, size, n, p):
    return maybe_resize(n * (1 - p) / p**2, size)


@_variance.register(NormalRV)
def normal_variance(op, rv, rng, size, mu, sigma):
    return maybe_resize(pt.broadcast_arrays(mu, sigma)[1] ** 2, size)


@_variance.register(ParetoRV)
def pareto_variance(op, rv, rng, size, alpha, m):
    return maybe_resize(
        pt.switch(alpha > 2, (m**2 * alpha) / ((alpha - 1) ** 2 * (alpha - 2)), np.nan), size
    )


@_variance.register(PoissonRV)
def poisson_variance(op, rv, rng, size, mu):
    return maybe_resize(mu, size)


@_variance.register(PolyaGammaRV)
def polya_gamma_variance(op, rv, rng, size, h, z):
    return maybe_resize(
        pt.switch(
            pt.eq(z, 0),
            h / 24,
            (h / (4 * z**3)) * (pt.sinh(z) - z) / pt.cosh(z / 2) ** 2,
        ),
        size,
    )


@_variance.register(RiceRV)
def rice_variance(op, rv, rng, size, b, sigma):
    # b is the shape parameter, nu = b * sigma is the noncentrality parameter
    nu = b * sigma
    nu_sigma_ratio = -(nu**2) / (2 * sigma**2)
    L_half = pt.exp(nu_sigma_ratio / 2) * (
        (1 - nu_sigma_ratio) * pt.i0(-nu_sigma_ratio / 2)
        - nu_sigma_ratio * pt.i1(-nu_sigma_ratio / 2)
    )
    return maybe_resize(2 * sigma**2 + nu**2 - (np.pi * sigma**2 / 2) * L_half**2, size)


@_variance.register(SkewNormalRV)
def skew_normal_variance(op, rv, rng, size, mu, sigma, alpha):
    delta = alpha / pt.sqrt(1 + alpha**2)
    return maybe_resize(sigma**2 * (1 - 2 * delta**2 / np.pi), size)


@_variance.register(SkewStudentTRV)
def skew_studentt_variance(op, rv, rng, size, a, b, mu, sigma):
    a, b, _, sigma = pt.broadcast_arrays(a, b, mu, sigma)
    mean_val = (
        (a - b)
        * pt.sqrt(a + b)
        * pt.gamma(a - 0.5)
        * pt.gamma(b - 0.5)
        / (2 * pt.gamma(a) * pt.gamma(b))
    )
    var = (a + b) / ((a - 1) * (b - 1)) - mean_val**2
    var = sigma**2 * var
    if not rv_size_is_none(size):
        var = pt.full(size, var)
    return var


@_variance.register(StickBreakingWeightsRV)
def stick_breaking_variance(op, rv, rng, size, alpha, K):
    raise NotImplementedError(
        "StickBreakingWeights variance not implemented due to complex covariance structure."
    )


@_variance.register(StudentTRV)
def studentt_variance(op, rv, rng, size, nu, mu, sigma):
    return maybe_resize(pt.switch(nu > 2, sigma**2 * nu / (nu - 2), np.nan), size)


@_variance.register(TriangularRV)
def triangular_variance(op, rv, rng, size, lower, c, upper):
    return maybe_resize(
        (lower**2 + upper**2 + c**2 - lower * upper - lower * c - upper * c) / 18, size
    )


@_variance.register(UniformRV)
def uniform_variance(op, rv, rng, size, lower, upper):
    return maybe_resize((upper - lower) ** 2 / 12, size)


@_variance.register(VonMisesRV)
def vonmises_variance(op, rv, rng, size, mu, kappa):
    return maybe_resize(1 - pt.i1(kappa) / pt.i0(kappa), size)


@_variance.register(WaldRV)
def wald_variance(op, rv, rng, size, mu, lam, _alpha):
    # _alpha is the shift parameter, which doesn't affect variance
    return maybe_resize(mu**3 / lam, size)


@_variance.register(WeibullBetaRV)
def weibull_variance(op, rv, rng, size, alpha, beta):
    return maybe_resize(beta**2 * (pt.gamma(1 + 2 / alpha) - pt.gamma(1 + 1 / alpha) ** 2), size)


@_variance.register(WishartRV)
def wishart_variance(op, rv, rng, size, nu, V):
    diag_V = pt.diag(V)
    var = nu * (V**2 + diag_V[:, None] * diag_V[None, :])
    if not rv_size_is_none(size):
        var_size = pt.concatenate([size, V.shape[-2:]])
        var = pt.full(var_size, var)
    return var
