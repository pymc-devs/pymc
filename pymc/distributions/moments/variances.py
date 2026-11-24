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
from functools import reduce, singledispatch

import numpy as np

from pytensor import sparse as ps
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
)
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable, Variable

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
from pymc.distributions.distribution import DiracDeltaRV, SymbolicRandomVariable
from pymc.distributions.mixture import MixtureRV
from pymc.distributions.moments.means import _mean, mean
from pymc.distributions.multivariate import (
    CARRV,
    DirichletMultinomialRV,
    KroneckerNormalRV,
    MvStudentTRV,
    WishartRV,
)
from pymc.exceptions import UndefinedMomentException

__all__ = ["variance"]


@singledispatch
def _variance(op, rv, *rv_inputs) -> TensorVariable:
    raise NotImplementedError(f"Variable {rv} of type {op} has no variance implementation.")


def variance(rv: TensorVariable) -> TensorVariable:
    """Compute the variance of each element of a random variable.

    A multivariate random variable gets the marginal variance of each element, not a covariance
    matrix. Integer parameters are cast to ``floatX`` before the variance is computed.

    Parameters
    ----------
    rv : TensorVariable
        The random variable.

    Returns
    -------
    variance : TensorVariable
        The variance of each element of ``rv``, with the shape of ``rv``.
    """
    node = rv.owner
    parameters = set(node.op.dist_params(node))
    inputs = [
        _as_float(variable) if variable in parameters and not _is_random(variable) else variable
        for variable in node.inputs
    ]
    return pt.broadcast_to(_variance(node.op, rv, *inputs), rv.shape)


def _as_float(variable: Variable) -> Variable:
    if np.dtype(variable.type.dtype).kind in "biu":
        return pt.cast(variable, "floatX")
    return variable


def _is_random(variable: Variable) -> bool:
    return variable.owner is not None and isinstance(
        variable.owner.op, RandomVariable | SymbolicRandomVariable
    )


@_variance.register(AsymmetricLaplaceRV)
def asymmetric_laplace_variance(op, rv, rng, size, b, kappa, mu):
    return (1 + kappa**4) / (b**2 * kappa**2)


@_variance.register(BernoulliRV)
def bernoulli_variance(op, rv, rng, size, p):
    return p * (1 - p)


@_variance.register(BetaRV)
def beta_variance(op, rv, rng, size, alpha, beta):
    alpha, beta = pt.cast(alpha, "floatX"), pt.cast(beta, "floatX")
    return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))


@_variance.register(BetaBinomialRV)
def betabinomial_variance(op, rv, rng, size, n, alpha, beta):
    n, alpha, beta = pt.cast(n, "floatX"), pt.cast(alpha, "floatX"), pt.cast(beta, "floatX")
    return (n * alpha * beta * (alpha + beta + n)) / ((alpha + beta) ** 2 * (alpha + beta + 1))


@_variance.register(BinomialRV)
def binomial_variance(op, rv, rng, size, n, p):
    return n * p * (1 - p)


@_variance.register(CARRV)
def car_variance(op, rv, rng, size, mu, W, alpha, tau, W_is_valid):
    if isinstance(W.type, ps.SparseTensorType):
        W = ps.dense_from_sparse(W)
    D = pt.sum(W, axis=-1)[..., :, None] * pt.eye(W.shape[-1])
    Q = tau[..., None, None] * (D - alpha[..., None, None] * W)
    return pt.diagonal(pt.linalg.inv(Q), axis1=-2, axis2=-1)


@_variance.register(CategoricalRV)
def categorical_variance(op, rv, *args):
    raise UndefinedMomentException("The variance of the Categorical distribution is undefined")


@_variance.register(CauchyRV)
def cauchy_variance(op, rv, rng, size, alpha, beta):
    raise UndefinedMomentException("The variance of the Cauchy distribution is undefined")


@_variance.register(DiracDeltaRV)
def dirac_delta_variance(op, rv, size, c):
    return pt.zeros_like(c)


@_variance.register(DirichletRV)
def dirichlet_variance(op, rv, rng, size, a):
    a0 = pt.sum(a, axis=-1, keepdims=True)
    return (a * (a0 - a)) / (a0**2 * (a0 + 1))


@_variance.register(DirichletMultinomialRV)
def dirichlet_multinomial_variance(op, rv, rng, size, n, a):
    a0 = pt.sum(a, axis=-1, keepdims=True)
    n_expanded = pt.shape_padright(n)
    return n_expanded * (a / a0) * (1 - a / a0) * (a0 + n_expanded) / (a0 + 1)


@_variance.register(DiscreteUniformRV)
def discrete_uniform_variance(op, rv, rng, size, lower, upper):
    return ((upper - lower + 1) ** 2 - 1) / 12.0


@_variance.register(ExGaussianRV)
def exgaussian_variance(op, rv, rng, size, mu, nu, sigma):
    _, nu, sigma = pt.broadcast_arrays(mu, nu, sigma)
    return sigma**2 + nu**2


@_variance.register(ExponentialRV)
def exponential_variance(op, rv, rng, size, mu):
    return mu**2


@_variance.register(FlatRV)
def flat_variance(op, rv, rng, size):
    raise UndefinedMomentException("The variance of Flat distribution is undefined")


@_variance.register(GammaRV)
def gamma_variance(op, rv, rng, size, alpha, inv_beta):
    # The pytensor `GammaRV` `Op` inverts the `beta` parameter itself
    return alpha * inv_beta**2


@_variance.register(GeometricRV)
def geometric_variance(op, rv, rng, size, p):
    return (1 - p) / p**2


@_variance.register(GumbelRV)
def gumbel_variance(op, rv, rng, size, mu, beta):
    return np.pi**2 / 6 * beta**2


@_variance.register(HalfStudentTRV)
def half_studentt_variance(op, rv, rng, size, nu, sigma):
    second_moment = sigma**2 * nu / (nu - 2)
    mean_val = _mean(op, rv, rng, size, nu, sigma)
    return pt.switch(nu > 2, second_moment - mean_val**2, np.inf)


@_variance.register(HalfCauchyRV)
def pymc_halfcauchy_variance(op, rv, rng, size, beta):
    raise UndefinedMomentException("The variance of the HalfCauchy distribution is undefined")


@_variance.register(HalfFlatRV)
def halfflat_variance(op, rv, rng, size):
    raise UndefinedMomentException("The variance of the HalfFlat distribution is undefined")


@_variance.register(HalfNormalRV)
def halfnormal_variance(op, rv, rng, size, loc, sigma):
    _, sigma = pt.broadcast_arrays(loc, sigma)
    return sigma**2 * (1 - 2 / np.pi)


@_variance.register(HyperGeometricRV)
def hypergeometric_variance(op, rv, rng, size, good, bad, n):
    N = good + bad
    K = good
    return n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))


@_variance.register(InvGammaRV)
def invgamma_variance(op, rv, rng, size, alpha, beta):
    return pt.switch(alpha > 2, beta**2 / ((alpha - 1) ** 2 * (alpha - 2)), np.inf)


@_variance.register(KroneckerNormalRV)
def kronecker_normal_variance(op, rv, rng, size, mu, sigma, *covs):
    # The diagonal of a Kronecker product is the Kronecker product of the diagonals.
    diagonals = [pt.diagonal(cov, axis1=-2, axis2=-1) for cov in covs]
    return reduce(lambda left, right: pt.outer(left, right).ravel(), diagonals) + sigma**2


@_variance.register(KumaraswamyRV)
def kumaraswamy_variance(op, rv, rng, size, a, b):
    second_moment = pt.exp(
        pt.log(b) + pt.gammaln(1 + 2 / a) + pt.gammaln(b) - pt.gammaln(1 + 2 / a + b)
    )
    return second_moment - _mean(op, rv, rng, size, a, b) ** 2


@_variance.register(LaplaceRV)
def laplace_variance(op, rv, rng, size, mu, b):
    return 2 * b**2


@_variance.register(LogisticRV)
def logistic_variance(op, rv, rng, size, mu, s):
    return np.pi**2 / 3 * s**2


@_variance.register(LogNormalRV)
def lognormal_variance(op, rv, rng, size, mu, sigma):
    return (pt.exp(sigma**2) - 1) * pt.exp(2 * mu + sigma**2)


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
        weights * (mean_components - pt.expand_dims(mixture_mean, mix_axis)) ** 2, axis=mix_axis
    )
    return mean_of_variance + variance_of_mean


@_variance.register(MoyalRV)
def moyal_variance(op, rv, rng, size, mu, sigma):
    return np.pi**2 / 2 * sigma**2


@_variance.register(MultinomialRV)
def multinomial_variance(op, rv, rng, size, n, p):
    return n[..., None] * p * (1 - p)


@_variance.register(MvNormalRV)
def mvnormal_variance(op, rv, rng, size, mu, cov):
    return pt.diagonal(cov, axis1=-2, axis2=-1)


@_variance.register(MvStudentTRV)
def mvstudentt_variance(op, rv, rng, size, nu, mu, scale):
    nu = nu[..., None]
    scale_variance = pt.diagonal(scale, axis1=-2, axis2=-1)
    return pt.switch(nu > 2, nu / (nu - 2) * scale_variance, pt.switch(nu > 1, np.inf, np.nan))


@_variance.register(NegBinomialRV)
def negative_binomial_variance(op, rv, rng, size, n, p):
    return n * (1 - p) / p**2


@_variance.register(NormalRV)
def normal_variance(op, rv, rng, size, mu, sigma):
    return pt.broadcast_arrays(mu, sigma)[1] ** 2


@_variance.register(ParetoRV)
def pareto_variance(op, rv, rng, size, alpha, m):
    return pt.switch(alpha > 2, (m**2 * alpha) / ((alpha - 1) ** 2 * (alpha - 2)), np.inf)


@_variance.register(PoissonRV)
def poisson_variance(op, rv, rng, size, mu):
    return mu


@_variance.register(PolyaGammaRV)
def polya_gamma_variance(op, rv, rng, size, h, z):
    return pt.switch(
        pt.eq(z, 0),
        h / 24,
        (h / (4 * z**3)) * (pt.sinh(z) - z) / pt.cosh(z / 2) ** 2,
    )


@_variance.register(RiceRV)
def rice_variance(op, rv, rng, size, b, sigma):
    second_moment = 2 * sigma**2 + (b * sigma) ** 2
    return second_moment - _mean(op, rv, rng, size, b, sigma) ** 2


@_variance.register(SkewNormalRV)
def skew_normal_variance(op, rv, rng, size, mu, sigma, alpha):
    delta = alpha / pt.sqrt(1 + alpha**2)
    return sigma**2 * (1 - 2 * delta**2 / np.pi)


@_variance.register(SkewStudentTRV)
def skew_studentt_variance(op, rv, rng, size, a, b, mu, sigma):
    second_moment = (a + b) / ((a - 1) * (b - 1)) * sigma**2
    centered_mean = _mean(op, rv, rng, size, a, b, mu, sigma) - mu
    return pt.switch(
        (a > 1) & (b > 1),
        second_moment - centered_mean**2,
        pt.switch((a > 0.5) & (b > 0.5), np.inf, np.nan),
    )


@_variance.register(StudentTRV)
def studentt_variance(op, rv, rng, size, nu, mu, sigma):
    return pt.switch(nu > 2, sigma**2 * nu / (nu - 2), pt.switch(nu > 1, np.inf, np.nan))


@_variance.register(TriangularRV)
def triangular_variance(op, rv, rng, size, lower, c, upper):
    return (lower**2 + upper**2 + c**2 - lower * upper - lower * c - upper * c) / 18


@_variance.register(UniformRV)
def uniform_variance(op, rv, rng, size, lower, upper):
    return (upper - lower) ** 2 / 12


@_variance.register(WaldRV)
def wald_variance(op, rv, rng, size, mu, lam, _alpha):
    # _alpha is the shift parameter, which doesn't affect variance
    return mu**3 / lam


@_variance.register(WeibullBetaRV)
def weibull_variance(op, rv, rng, size, alpha, beta):
    return beta**2 * (pt.gamma(1 + 2 / alpha) - pt.gamma(1 + 1 / alpha) ** 2)


@_variance.register(WishartRV)
def wishart_variance(op, rv, rng, size, nu, scale_chol):
    V = scale_chol @ scale_chol.mT
    diag_V = pt.diagonal(V, axis1=-2, axis2=-1)
    return nu[..., None, None] * (V**2 + diag_V[..., :, None] * diag_V[..., None, :])
