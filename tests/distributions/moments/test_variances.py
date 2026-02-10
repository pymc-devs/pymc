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

import warnings

import numpy as np
import pytest

from scipy.stats import (
    bernoulli,
    beta,
    betabinom,
    binom,
    chi2,
    dirichlet,
    expon,
    exponnorm,
    gamma,
    geom,
    gumbel_r,
    halfnorm,
    hypergeom,
    invgamma,
    invgauss,
    jf_skew_t,
    laplace,
    laplace_asymmetric,
    logistic,
    lognorm,
    moyal,
    multinomial,
    nbinom,
    norm,
    pareto,
    poisson,
    rice,
    skewnorm,
    t,
    triang,
    uniform,
    weibull_min,
    wishart,
)

from pymc import (
    CAR,
    AsymmetricLaplace,
    Bernoulli,
    Beta,
    BetaBinomial,
    Binomial,
    Categorical,
    Cauchy,
    ChiSquared,
    DiracDelta,
    Dirichlet,
    DirichletMultinomial,
    DiscreteUniform,
    ExGaussian,
    Exponential,
    Flat,
    Gamma,
    Geometric,
    Gumbel,
    HalfCauchy,
    HalfFlat,
    HalfNormal,
    HalfStudentT,
    HyperGeometric,
    InverseGamma,
    KroneckerNormal,
    Kumaraswamy,
    Laplace,
    LKJCholeskyCov,
    LKJCorr,
    Logistic,
    LogitNormal,
    LogNormal,
    MatrixNormal,
    Mixture,
    Moyal,
    Multinomial,
    MvNormal,
    MvStudentT,
    NegativeBinomial,
    Normal,
    Pareto,
    Poisson,
    PolyaGamma,
    Rice,
    SkewNormal,
    SkewStudentT,
    StickBreakingWeights,
    StudentT,
    Triangular,
    Uniform,
    VonMises,
    Wald,
    Weibull,
    ZeroInflatedBinomial,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
from pymc.distributions.moments.variances import variance
from pymc.exceptions import UndefinedMomentException


def dirichlet_multinomial_variance(n: int, a: np.ndarray) -> np.ndarray:
    """Compute DirichletMultinomial variance: n * p * (1-p) * (n + sum(a)) / (sum(a) + 1)."""
    a0 = a.sum()
    p = a / a0
    return n * p * (1 - p) * (n + a0) / (a0 + 1)


def assert_batched_variance_consistent(dist, dist_params, base_variance):
    """Check that batched RVs produce correctly tiled variances."""
    rv_batched = dist.dist(shape=(3, *base_variance.shape), **dist_params)
    batched_var = variance(rv_batched).eval()
    expected = np.tile(base_variance, (3,) + (1,) * base_variance.ndim)
    np.testing.assert_almost_equal(batched_var, expected)


@pytest.mark.parametrize(
    ["dist", "scipy_equiv", "dist_params", "scipy_params"],
    [
        [
            AsymmetricLaplace,
            laplace_asymmetric,
            {"kappa": 2, "mu": 0.2, "b": 1 / 1.2},
            {"kappa": 2, "loc": 0.2, "scale": 1.2},
        ],
        [Bernoulli, bernoulli, {"p": 0.6}, {"p": 0.6}],
        [Beta, beta, {"alpha": 3, "beta": 2}, {"a": 3, "b": 2}],
        [BetaBinomial, betabinom, {"alpha": 3, "beta": 2, "n": 5}, {"a": 3, "b": 2, "n": 5}],
        [Binomial, binom, {"p": 0.6, "n": 5}, {"p": 0.6, "n": 5}],
        [ChiSquared, chi2, {"nu": 6}, {"df": 6}],
        [Dirichlet, dirichlet, {"a": np.ones(4)}, {"alpha": np.ones(4)}],
        [ExGaussian, exponnorm, {"mu": 0, "sigma": 1, "nu": 1}, {"loc": 0, "scale": 1, "K": 1}],
        [Exponential, expon, {"lam": 1}, {"scale": 1}],
        [Gamma, gamma, {"alpha": 4, "beta": 3}, {"a": 4, "scale": 1 / 3}],
        [Geometric, geom, {"p": 0.1}, {"p": 0.1}],
        [Gumbel, gumbel_r, {"mu": 2, "beta": 1}, {"loc": 2, "scale": 1}],
        [HalfNormal, halfnorm, {"sigma": 1}, {"scale": 1}],
        [HyperGeometric, hypergeom, {"N": 10, "k": 2, "n": 4}, {"M": 10, "n": 2, "N": 4}],
        [InverseGamma, invgamma, {"alpha": 3, "beta": 2}, {"a": 3, "scale": 2}],
        [Laplace, laplace, {"mu": 2, "b": 2}, {"loc": 2, "scale": 2}],
        [Logistic, logistic, {"mu": 2, "s": 1}, {"loc": 2, "scale": 1}],
        [LogNormal, lognorm, {"mu": 0.3, "sigma": 0.6}, {"scale": np.exp(0.3), "s": 0.6}],
        [Moyal, moyal, {"mu": 2, "sigma": 2}, {"loc": 2, "scale": 2}],
        [NegativeBinomial, nbinom, {"n": 10, "p": 0.5}, {"n": 10, "p": 0.5}],
        [Normal, norm, {"mu": 2, "sigma": 2}, {"loc": 2, "scale": 2}],
        [Pareto, pareto, {"alpha": 5, "m": 2}, {"b": 5, "scale": 2}],
        [Poisson, poisson, {"mu": 20}, {"mu": 20}],
        [Rice, rice, {"b": 1, "sigma": 2}, {"b": 1, "scale": 2}],
        [SkewNormal, skewnorm, {"mu": 2, "sigma": 2, "alpha": 2}, {"loc": 2, "scale": 2, "a": 2}],
        [
            SkewStudentT,
            jf_skew_t,
            {"mu": 2, "sigma": 2, "a": 3, "b": 3},
            {"loc": 2, "scale": 2, "a": 3, "b": 3},
        ],
        [StudentT, t, {"mu": 2, "sigma": 2, "nu": 6}, {"loc": 2, "scale": 2, "df": 6}],
        [
            Triangular,
            triang,
            {"lower": -3, "upper": 2, "c": 1},
            {"loc": -3, "scale": 5, "c": 4 / 5},
        ],
        [Uniform, uniform, {"lower": -3, "upper": 2}, {"loc": -3, "scale": 5}],
        [Wald, invgauss, {"mu": 2, "lam": 1}, {"mu": 2, "scale": 1}],
        [Weibull, weibull_min, {"alpha": 2, "beta": 2}, {"c": 2, "scale": 2}],
    ],
)
def test_univariate_variance_equal_to_scipy(dist, scipy_equiv, dist_params, scipy_params):
    rv = dist.dist(**dist_params)
    pymc_var = variance(rv).eval()
    scipy_var = scipy_equiv(**scipy_params).var()

    assert np.asarray(pymc_var).shape == np.asarray(scipy_var).shape
    np.testing.assert_almost_equal(pymc_var, scipy_var)
    assert_batched_variance_consistent(dist, dist_params, pymc_var)


COV_3x3 = np.array([[2.0, 0.5, -0.3], [0.5, 1.5, 0.2], [-0.3, 0.2, 1.0]])


@pytest.mark.parametrize(
    ["dist", "dist_params", "expected_cov", "support_dim"],
    [
        [
            Multinomial,
            {"n": 20, "p": np.ones(6) / 6},
            lambda: multinomial(n=20, p=np.ones(6) / 6).cov(),
            6,
        ],
        [
            MvNormal,
            {"mu": np.ones(3), "cov": COV_3x3},
            lambda: COV_3x3,
            3,
        ],
        [
            MvStudentT,
            {"mu": np.ones(3), "cov": COV_3x3, "nu": 4},
            lambda: COV_3x3 * 4 / (4 - 2),
            3,
        ],
    ],
)
def test_multivariate_variance(dist, dist_params, expected_cov, support_dim):
    rv = dist.dist(**dist_params)
    pymc_var = variance(rv).eval()
    scipy_var = expected_cov()

    assert np.asarray(pymc_var).shape == np.asarray(scipy_var).shape
    np.testing.assert_almost_equal(pymc_var, scipy_var)

    rv_batched = dist.dist(shape=(3, support_dim), **dist_params)
    pymc_var_batched = variance(rv_batched).eval()
    expected_shape = (3, *pymc_var.shape)
    assert pymc_var_batched.shape == expected_shape
    np.testing.assert_almost_equal(pymc_var_batched, np.broadcast_to(pymc_var, expected_shape))


@pytest.mark.parametrize(
    ["dist", "dist_params", "expected"],
    [
        [DiracDelta, {"c": 4.0}, 0.0],
        [
            DirichletMultinomial,
            {"n": 5, "a": np.ones(5)},
            dirichlet_multinomial_variance(n=5, a=np.ones(5)),
        ],
        [DiscreteUniform, {"lower": 3, "upper": 5}, ((5 - 3 + 1) ** 2 - 1) / 12.0],
        [Kumaraswamy, {"a": 1, "b": 1}, 1 / 12],
        [Mixture, {"w": [0.5, 0.5], "comp_dists": Normal.dist(mu=[0, 0], sigma=[1, 2])}, 2.5],
        [PolyaGamma, {"h": 1, "z": 0}, 1 / 24],
    ],
)
def test_variance_equal_expected(dist, dist_params, expected):
    rv = dist.dist(**dist_params)
    pymc_var = variance(rv).eval()
    np.testing.assert_almost_equal(pymc_var, np.asarray(expected))
    assert_batched_variance_consistent(dist, dist_params, pymc_var)


def test_car_variance():
    from scipy.stats import multivariate_normal

    W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    alpha = 0.5
    tau = 2.0
    mu = np.zeros(3)

    rv = CAR.dist(mu=mu, W=W, alpha=alpha, tau=tau)
    pymc_var = variance(rv).eval()

    # Definition of CAR covariance matrix
    D = np.diag(W.sum(axis=1))
    Q = tau * (D - alpha * W)
    car_cov = np.linalg.inv(Q)

    mvn = multivariate_normal(mean=mu, cov=car_cov)
    expected = mvn.cov

    np.testing.assert_almost_equal(pymc_var, expected)


def test_kronecker_normal_variance():
    from functools import reduce

    from scipy.stats import multivariate_normal

    K1 = np.array([[2.0, 0.5], [0.5, 1.5]])
    K2 = np.array([[1.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 1.0]])
    sigma = 0.1
    mu = np.zeros(6)

    rv = KroneckerNormal.dist(mu=mu, covs=[K1, K2], sigma=sigma)
    pymc_var = variance(rv).eval()

    expected_cov = reduce(np.kron, [K1, K2]) + sigma**2 * np.eye(6)
    mvn = multivariate_normal(mean=mu, cov=expected_cov)

    np.testing.assert_almost_equal(pymc_var, mvn.cov)


def test_wishart_variance():
    nu = 10
    V = np.array([[2.0, 0.5], [0.5, 1.5]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pymc import Wishart

        rv = Wishart.dist(nu=nu, V=V)

    pymc_var = variance(rv).eval()
    scipy_var = wishart(df=nu, scale=V).var()

    np.testing.assert_almost_equal(pymc_var, scipy_var)


@pytest.mark.parametrize(
    ["dist", "dist_params"],
    [
        [HalfStudentT, {"nu": 10, "sigma": 2}],
        [VonMises, {"mu": 2, "kappa": 2}],
        [ZeroInflatedBinomial, {"n": 10, "p": 0.5, "psi": 0.8}],
        [ZeroInflatedNegativeBinomial, {"n": 10, "p": 0.5, "psi": 0.8}],
        [ZeroInflatedPoisson, {"mu": 5, "psi": 0.8}],
    ],
)
def test_variance_computes_without_error(dist, dist_params):
    """Smoke test for distributions without closed-form expected values."""
    rv = dist.dist(**dist_params)
    pymc_var = variance(rv).eval()
    assert_batched_variance_consistent(dist, dist_params, pymc_var)


@pytest.mark.parametrize(
    ["dist", "dist_params"],
    [
        [Cauchy, {"alpha": 1, "beta": 1}],
        [HalfCauchy, {"beta": 1.0}],
        [Flat, {}],
        [HalfFlat, {}],
    ],
)
def test_variance_infinite(dist, dist_params):
    """Test that distributions with infinite variance return inf."""
    rv = dist.dist(**dist_params)
    pymc_var = variance(rv).eval()
    assert np.isinf(pymc_var)


@pytest.mark.parametrize(
    ["dist", "dist_params"],
    [
        [LogitNormal, {"mu": 2, "sigma": 1}],
        [Categorical, {"p": [0.1, 0.9]}],
        [
            LKJCholeskyCov,
            {"eta": 1, "n": 3, "sd_dist": DiracDelta.dist(1), "compute_corr": False},
        ],
        [LKJCorr, {"eta": 1, "n": 3}],
        [
            MatrixNormal,
            {"mu": np.eye(3), "rowcov": np.eye(3), "colcov": np.eye(3)},
        ],
        [StickBreakingWeights, {"alpha": 1, "K": 5}],
    ],
)
def test_variance_raises_for_undefined_distributions(dist, dist_params):
    with pytest.raises((UndefinedMomentException, NotImplementedError)):
        variance(dist.dist(**dist_params))
