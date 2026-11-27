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


import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from scipy import sparse
from scipy.special import gammaln
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
    Wishart,
    ZeroInflatedBinomial,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
from pymc import draw as pm_draw
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
        [InverseGamma, invgamma, {"alpha": 1.5, "beta": 2}, {"a": 1.5, "scale": 2}],
        [Laplace, laplace, {"mu": 2, "b": 2}, {"loc": 2, "scale": 2}],
        [Logistic, logistic, {"mu": 2, "s": 1}, {"loc": 2, "scale": 1}],
        [LogNormal, lognorm, {"mu": 0.3, "sigma": 0.6}, {"scale": np.exp(0.3), "s": 0.6}],
        [Moyal, moyal, {"mu": 2, "sigma": 2}, {"loc": 2, "scale": 2}],
        [NegativeBinomial, nbinom, {"n": 10, "p": 0.5}, {"n": 10, "p": 0.5}],
        [Normal, norm, {"mu": 2, "sigma": 2}, {"loc": 2, "scale": 2}],
        [Normal, norm, {"mu": 0, "sigma": 12}, {"loc": 0, "scale": 12}],
        [Pareto, pareto, {"alpha": 5, "m": 2}, {"b": 5, "scale": 2}],
        [Pareto, pareto, {"alpha": 1.5, "m": 2}, {"b": 1.5, "scale": 2}],
        [Poisson, poisson, {"mu": 20}, {"mu": 20}],
        [Rice, rice, {"b": 1, "sigma": 2}, {"b": 1, "scale": 2}],
        [Rice, rice, {"b": 15, "sigma": 1}, {"b": 15, "scale": 1}],
        [SkewNormal, skewnorm, {"mu": 2, "sigma": 2, "alpha": 2}, {"loc": 2, "scale": 2, "a": 2}],
        [
            SkewStudentT,
            jf_skew_t,
            {"mu": 2, "sigma": 2, "a": 3, "b": 3},
            {"loc": 2, "scale": 2, "a": 3, "b": 3},
        ],
        [StudentT, t, {"mu": 2, "sigma": 2, "nu": 6}, {"loc": 2, "scale": 2, "df": 6}],
        [StudentT, t, {"mu": 2, "sigma": 2, "nu": 1.5}, {"loc": 2, "scale": 2, "df": 1.5}],
        [StudentT, t, {"mu": 2, "sigma": 2, "nu": 0.8}, {"loc": 2, "scale": 2, "df": 0.8}],
        [
            Triangular,
            triang,
            {"lower": -3, "upper": 2, "c": 1},
            {"loc": -3, "scale": 5, "c": 4 / 5},
        ],
        [Uniform, uniform, {"lower": -3, "upper": 2}, {"loc": -3, "scale": 5}],
        [Uniform, uniform, {"lower": 0, "upper": 20}, {"loc": 0, "scale": 20}],
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
    ["dist", "dist_params", "expected_variance", "support_dim"],
    [
        [
            Multinomial,
            {"n": 20, "p": np.ones(6) / 6},
            np.diag(multinomial(n=20, p=np.ones(6) / 6).cov()),
            6,
        ],
        [MvNormal, {"mu": np.ones(3), "cov": COV_3x3}, np.diag(COV_3x3), 3],
        [MvStudentT, {"mu": np.ones(3), "scale": COV_3x3, "nu": 4}, np.diag(COV_3x3) * 4 / 2, 3],
        [MvStudentT, {"mu": np.ones(3), "scale": COV_3x3, "nu": 1.5}, np.full(3, np.inf), 3],
    ],
)
def test_multivariate_variance(dist, dist_params, expected_variance, support_dim):
    rv = dist.dist(**dist_params)
    pymc_var = variance(rv).eval()

    np.testing.assert_almost_equal(pymc_var, expected_variance)

    rv_batched = dist.dist(shape=(3, support_dim), **dist_params)
    np.testing.assert_almost_equal(
        variance(rv_batched).eval(), np.broadcast_to(expected_variance, (3, support_dim))
    )


WISHART_SCALES = np.stack(
    [np.array([[2.0, 0.5], [0.5, 1.5]]), np.array([[1.0, -0.2], [-0.2, 3.0]])]
)
CAR_W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)


@pytest.mark.parametrize(
    ["dist", "dist_params", "expected_variance"],
    [
        [
            MvNormal,
            {"mu": np.ones((4, 3)), "cov": COV_3x3},
            np.broadcast_to(np.diag(COV_3x3), (4, 3)),
        ],
        [
            MvStudentT,
            {"mu": np.ones(3), "scale": COV_3x3, "nu": np.array([3.0, 4.0, 6.0])},
            (np.array([3.0, 4.0, 6.0]) / np.array([1.0, 2.0, 4.0]))[:, None] * np.diag(COV_3x3),
        ],
        [
            Wishart,
            {"nu": 10, "V": WISHART_SCALES},
            [wishart(df=10, scale=scale).var() for scale in WISHART_SCALES],
        ],
        [
            CAR,
            {"mu": np.zeros(3), "W": CAR_W, "alpha": 0.5, "tau": 2.0, "size": (4,)},
            np.broadcast_to(
                np.diag(np.linalg.inv(2.0 * (np.diag(CAR_W.sum(axis=1)) - 0.5 * CAR_W))), (4, 3)
            ),
        ],
    ],
)
def test_multivariate_variance_with_batched_parameters(dist, dist_params, expected_variance):
    rv = dist.dist(**dist_params)
    np.testing.assert_almost_equal(variance(rv).eval(), expected_variance)


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
        [
            HalfStudentT,
            {"nu": 10, "sigma": 2},
            4 * 10 / 8
            - (2 * 2 * np.sqrt(10 / np.pi) * np.exp(gammaln(5.5) - gammaln(5) - np.log(9))) ** 2,
        ],
        [SkewStudentT, {"mu": 0, "sigma": 1, "a": 0.8, "b": 3}, np.inf],
        [SkewStudentT, {"mu": 0, "sigma": 1, "a": 0.4, "b": 3}, np.nan],
        [ZeroInflatedPoisson, {"mu": 5, "psi": 0.8}, 0.8 * 5 * (1 + 0.2 * 5)],
        [
            ZeroInflatedBinomial,
            {"n": 10, "p": 0.5, "psi": 0.8},
            0.8 * 10 * 0.5 * 0.5 + 0.8 * 0.2 * (10 * 0.5) ** 2,
        ],
        [
            ZeroInflatedNegativeBinomial,
            {"n": 10, "p": 0.5, "psi": 0.8},
            0.8 * 10 * 0.5 / 0.5**2 + 0.8 * 0.2 * (10 * 0.5 / 0.5) ** 2,
        ],
    ],
)
def test_variance_equal_expected(dist, dist_params, expected):
    rv = dist.dist(**dist_params)
    pymc_var = variance(rv).eval()
    np.testing.assert_almost_equal(pymc_var, np.asarray(expected))
    assert_batched_variance_consistent(dist, dist_params, pymc_var)


@pytest.mark.parametrize("to_sparse", [False, True], ids=["dense", "sparse"])
def test_car_variance(to_sparse):
    W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    rv = CAR.dist(mu=np.zeros(3), W=sparse.csr_matrix(W) if to_sparse else W, alpha=0.5, tau=2.0)

    # A check against the formula: CAR's sampler draws with covariance Q^-2.
    precision = 2.0 * (np.diag(W.sum(axis=1)) - 0.5 * W)
    np.testing.assert_almost_equal(variance(rv).eval(), np.diag(np.linalg.inv(precision)))


def test_kronecker_normal_variance_matches_draws():
    K1 = np.array([[2.0, 0.5], [0.5, 1.5]])
    K2 = np.array([[1.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 1.0]])
    rv = KroneckerNormal.dist(mu=np.zeros(6), covs=[K1, K2], sigma=0.1)

    draws = pm_draw(rv, draws=20_000, random_seed=1)
    np.testing.assert_allclose(variance(rv).eval(), draws.var(axis=0), rtol=0.05)


def test_multivariate_mixture_variance_matches_draws():
    components = [
        MvNormal.dist(mu=np.array([0.0, 1.0, -1.0]), cov=COV_3x3),
        MvNormal.dist(mu=np.array([2.0, -1.0, 0.5]), cov=0.5 * np.eye(3)),
    ]
    rv = Mixture.dist(w=[0.3, 0.7], comp_dists=components)

    draws = pm_draw(rv, draws=40_000, random_seed=1)
    np.testing.assert_allclose(variance(rv).eval(), draws.var(axis=0), rtol=0.05)


def test_wishart_variance():
    V = np.array([[2.0, 0.5], [0.5, 1.5]])
    rv = Wishart.dist(nu=10, V=V)

    np.testing.assert_almost_equal(variance(rv).eval(), wishart(df=10, scale=V).var())


@pytest.mark.parametrize(
    ["dist", "dist_params"],
    [
        [Categorical, {"p": [0.1, 0.9]}],
        [Cauchy, {"alpha": 1, "beta": 1}],
        [HalfCauchy, {"beta": 1.0}],
        [Flat, {}],
        [HalfFlat, {}],
    ],
)
def test_variance_undefined(dist, dist_params):
    rv = dist.dist(**dist_params)
    with pytest.raises(UndefinedMomentException, match="undefined"):
        variance(rv)


@pytest.mark.parametrize(
    ["dist", "dist_params"],
    [
        [LogitNormal, {"mu": 2, "sigma": 1}],
        [
            LKJCholeskyCov,
            {"eta": 1, "n": 3, "sd_dist": DiracDelta.dist(1), "compute_corr": False},
        ],
        [LKJCorr, {"eta": 1, "n": 3}],
        [MatrixNormal, {"mu": np.eye(3), "rowcov": np.eye(3), "colcov": np.eye(3)}],
        [StickBreakingWeights, {"alpha": 1, "K": 5}],
        [VonMises, {"mu": 2, "kappa": 2}],
    ],
)
def test_variance_not_implemented(dist, dist_params):
    with pytest.raises(NotImplementedError):
        variance(dist.dist(**dist_params))


@pytest.mark.parametrize(
    ["dist", "parameter", "value", "fixed_params"],
    [
        [StudentT, "nu", 5.0, {"mu": 0.0, "sigma": 2.0}],
        [Rice, "b", 3.0, {"sigma": 1.5}],
        [SkewStudentT, "a", 3.0, {"b": 2.5, "mu": 0.0, "sigma": 1.0}],
    ],
)
def test_variance_gradient_matches_finite_differences(dist, parameter, value, fixed_params):
    symbolic = pt.scalar(parameter)
    var = variance(dist.dist(**{parameter: symbolic}, **fixed_params))
    var_and_gradient = pytensor.function([symbolic], [var, pytensor.grad(var, symbolic)])

    step = 1e-6
    finite_difference = (var_and_gradient(value + step)[0] - var_and_gradient(value - step)[0]) / (
        2 * step
    )
    np.testing.assert_allclose(var_and_gradient(value)[1], finite_difference, rtol=1e-5)
