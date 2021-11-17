import aesara
import numpy as np
import pytest

from aesara import tensor as at
from scipy import special

import pymc as pm

from pymc.distributions import (
    AsymmetricLaplace,
    Bernoulli,
    Beta,
    BetaBinomial,
    Binomial,
    Categorical,
    Cauchy,
    ChiSquared,
    Constant,
    DensityDist,
    Dirichlet,
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
    Kumaraswamy,
    Laplace,
    Logistic,
    LogitNormal,
    LogNormal,
    Moyal,
    NegativeBinomial,
    Normal,
    Pareto,
    Poisson,
    Rice,
    SkewNormal,
    StudentT,
    Triangular,
    TruncatedNormal,
    Uniform,
    Wald,
    Weibull,
    ZeroInflatedBinomial,
    ZeroInflatedPoisson,
)
from pymc.distributions.distribution import get_moment
from pymc.distributions.multivariate import MvNormal
from pymc.distributions.shape_utils import rv_size_is_none, to_tuple
from pymc.initial_point import make_initial_point_fn
from pymc.model import Model


def test_rv_size_is_none():
    rv = Normal.dist(0, 1, size=None)
    assert rv_size_is_none(rv.owner.inputs[1])

    rv = Normal.dist(0, 1, size=1)
    assert not rv_size_is_none(rv.owner.inputs[1])

    size = Bernoulli.dist(0.5)
    rv = Normal.dist(0, 1, size=size)
    assert not rv_size_is_none(rv.owner.inputs[1])

    size = Normal.dist(0, 1).size
    rv = Normal.dist(0, 1, size=size)
    assert not rv_size_is_none(rv.owner.inputs[1])


def assert_moment_is_expected(model, expected):
    fn = make_initial_point_fn(
        model=model,
        return_transformed=False,
        default_strategy="moment",
    )
    result = fn(0)["x"]
    expected = np.asarray(expected)
    try:
        random_draw = model["x"].eval()
    except NotImplementedError:
        random_draw = result
    assert result.shape == expected.shape == random_draw.shape
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "size, expected",
    [
        (None, 0),
        (5, np.zeros(5)),
        ((2, 5), np.zeros((2, 5))),
    ],
)
def test_flat_moment(size, expected):
    with Model() as model:
        Flat("x", size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "size, expected",
    [
        (None, 1),
        (5, np.ones(5)),
        ((2, 5), np.ones((2, 5))),
    ],
)
def test_halfflat_moment(size, expected):
    with Model() as model:
        HalfFlat("x", size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "lower, upper, size, expected",
    [
        (-1, 1, None, 0),
        (-1, 1, 5, np.zeros(5)),
        (0, np.arange(1, 6), None, np.arange(1, 6) / 2),
        (0, np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(1, 6) / 2)),
    ],
)
def test_uniform_moment(lower, upper, size, expected):
    with Model() as model:
        Uniform("x", lower=lower, upper=upper, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, sigma, size, expected",
    [
        (0, 1, None, 0),
        (0, np.ones(5), None, np.zeros(5)),
        (np.arange(5), 1, None, np.arange(5)),
        (np.arange(5), np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(5))),
    ],
)
def test_normal_moment(mu, sigma, size, expected):
    with Model() as model:
        Normal("x", mu=mu, sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "sigma, size, expected",
    [
        (1, None, 1),
        (1, 5, np.ones(5)),
        (np.arange(5), None, np.arange(5)),
        (np.arange(5), (2, 5), np.full((2, 5), np.arange(5))),
    ],
)
def test_halfnormal_moment(sigma, size, expected):
    with Model() as model:
        HalfNormal("x", sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "nu, sigma, size, expected",
    [
        (1, 1, None, 1),
        (1, 1, 5, np.ones(5)),
        (1, np.arange(5), (2, 5), np.full((2, 5), np.arange(5))),
        (np.arange(1, 6), 1, None, np.full(5, 1)),
    ],
)
def test_halfstudentt_moment(nu, sigma, size, expected):
    with Model() as model:
        HalfStudentT("x", nu=nu, sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.skip(reason="aeppl interval transform fails when both edges are None")
@pytest.mark.parametrize(
    "mu, sigma, lower, upper, size, expected",
    [
        (0.9, 1, -1, 1, None, 0),
        (0.9, 1, -np.inf, np.inf, 5, np.full(5, 0.9)),
        (np.arange(5), 1, None, 10, (2, 5), np.full((2, 5), 9)),
        (1, np.ones(5), -10, np.inf, None, np.full((2, 5), -9)),
    ],
)
def test_truncatednormal_moment(mu, sigma, lower, upper, size, expected):
    with Model() as model:
        TruncatedNormal("x", mu=mu, sigma=sigma, lower=lower, upper=upper, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "p, size, expected",
    [
        (0.3, None, 0),
        (0.9, 5, np.ones(5)),
        (np.linspace(0, 1, 4), None, [0, 0, 1, 1]),
        (np.linspace(0, 1, 4), (2, 4), np.full((2, 4), [0, 0, 1, 1])),
    ],
)
def test_bernoulli_moment(p, size, expected):
    with Model() as model:
        Bernoulli("x", p=p, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "alpha, beta, size, expected",
    [
        (1, 1, None, 0.5),
        (1, 1, 5, np.full(5, 0.5)),
        (1, np.arange(1, 6), None, 1 / np.arange(2, 7)),
        (1, np.arange(1, 6), (2, 5), np.full((2, 5), 1 / np.arange(2, 7))),
    ],
)
def test_beta_moment(alpha, beta, size, expected):
    with Model() as model:
        Beta("x", alpha=alpha, beta=beta, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "n, alpha, beta, size, expected",
    [
        (10, 1, 1, None, 5),
        (10, 1, 1, 5, np.full(5, 5)),
        (10, 1, np.arange(1, 6), None, np.round(10 / np.arange(2, 7))),
        (10, 1, np.arange(1, 6), (2, 5), np.full((2, 5), np.round(10 / np.arange(2, 7)))),
    ],
)
def test_beta_binomial_moment(alpha, beta, n, size, expected):
    with Model() as model:
        BetaBinomial("x", alpha=alpha, beta=beta, n=n, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "nu, size, expected",
    [
        (1, None, 1),
        (1, 5, np.full(5, 1)),
        (np.arange(1, 6), None, np.arange(1, 6)),
    ],
)
def test_chisquared_moment(nu, size, expected):
    with Model() as model:
        ChiSquared("x", nu=nu, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "lam, size, expected",
    [
        (2, None, 0.5),
        (2, 5, np.full(5, 0.5)),
        (np.arange(1, 5), None, 1 / np.arange(1, 5)),
        (np.arange(1, 5), (2, 4), np.full((2, 4), 1 / np.arange(1, 5))),
    ],
)
def test_exponential_moment(lam, size, expected):
    with Model() as model:
        Exponential("x", lam=lam, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, b, size, expected",
    [
        (0, 1, None, 0),
        (0, np.ones(5), None, np.zeros(5)),
        (np.arange(5), 1, None, np.arange(5)),
        (np.arange(5), np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(5))),
    ],
)
def test_laplace_moment(mu, b, size, expected):
    with Model() as model:
        Laplace("x", mu=mu, b=b, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, nu, sigma, size, expected",
    [
        (0, 1, 1, None, 0),
        (0, np.ones(5), 1, None, np.zeros(5)),
        (np.arange(5), 10, np.arange(1, 6), None, np.arange(5)),
        (
            np.arange(5),
            10,
            np.arange(1, 6),
            (2, 5),
            np.full((2, 5), np.arange(5)),
        ),
    ],
)
def test_studentt_moment(mu, nu, sigma, size, expected):
    with Model() as model:
        StudentT("x", mu=mu, nu=nu, sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "alpha, beta, size, expected",
    [
        (0, 1, None, 0),
        (0, np.ones(5), None, np.zeros(5)),
        (np.arange(5), 1, None, np.arange(5)),
        (np.arange(5), np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(5))),
    ],
)
def test_cauchy_moment(alpha, beta, size, expected):
    with Model() as model:
        Cauchy("x", alpha=alpha, beta=beta, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "a, b, size, expected",
    [
        (1, 1, None, 0.5),
        (1, 1, 5, np.full(5, 0.5)),
        (1, np.arange(1, 6), None, 1 / np.arange(2, 7)),
        (np.arange(1, 6), 1, None, np.arange(1, 6) / np.arange(2, 7)),
        (1, np.arange(1, 6), (2, 5), np.full((2, 5), 1 / np.arange(2, 7))),
    ],
)
def test_kumaraswamy_moment(a, b, size, expected):
    with Model() as model:
        Kumaraswamy("x", a=a, b=b, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, sigma, size, expected",
    [
        (0, 1, None, np.exp(0.5)),
        (0, 1, 5, np.full(5, np.exp(0.5))),
        (np.arange(5), 1, None, np.exp(np.arange(5) + 0.5)),
        (
            np.arange(5),
            np.arange(1, 6),
            (2, 5),
            np.full((2, 5), np.exp(np.arange(5) + 0.5 * np.arange(1, 6) ** 2)),
        ),
    ],
)
def test_lognormal_moment(mu, sigma, size, expected):
    with Model() as model:
        LogNormal("x", mu=mu, sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "beta, size, expected",
    [
        (1, None, 1),
        (1, 5, np.ones(5)),
        (np.arange(5), None, np.arange(5)),
        (
            np.arange(5),
            (2, 5),
            np.full((2, 5), np.arange(5)),
        ),
    ],
)
def test_halfcauchy_moment(beta, size, expected):
    with Model() as model:
        HalfCauchy("x", beta=beta, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "alpha, beta, size, expected",
    [
        (1, 1, None, 1),
        (1, 1, 5, np.full(5, 1)),
        (np.arange(1, 6), 1, None, np.arange(1, 6)),
        (
            np.arange(1, 6),
            2 * np.arange(1, 6),
            (2, 5),
            np.full((2, 5), 0.5),
        ),
    ],
)
def test_gamma_moment(alpha, beta, size, expected):
    with Model() as model:
        Gamma("x", alpha=alpha, beta=beta, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "alpha, m, size, expected",
    [
        (2, 1, None, 1 * 2 ** (1 / 2)),
        (2, 1, 5, np.full(5, 1 * 2 ** (1 / 2))),
        (np.arange(2, 7), np.arange(1, 6), None, np.arange(1, 6) * 2 ** (1 / np.arange(2, 7))),
        (
            np.arange(2, 7),
            np.arange(1, 6),
            (2, 5),
            np.full((2, 5), np.arange(1, 6) * 2 ** (1 / np.arange(2, 7))),
        ),
    ],
)
def test_pareto_moment(alpha, m, size, expected):
    with Model() as model:
        Pareto("x", alpha=alpha, m=m, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, lam, phi, size, expected",
    [
        (2, None, None, None, 2),
        (None, 1, 1, 5, np.full(5, 1)),
        (1, None, np.ones(5), None, np.full(5, 1)),
        (3, np.full(5, 2), None, None, np.full(5, 3)),
        (np.arange(1, 6), None, np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(1, 6))),
    ],
)
def test_wald_moment(mu, lam, phi, size, expected):
    with Model() as model:
        Wald("x", mu=mu, lam=lam, phi=phi, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "alpha, beta, size, expected",
    [
        (1, 1, None, 1),
        (1, 1, 5, np.full(5, 1)),
        (np.arange(1, 6), 1, None, special.gamma(1 + 1 / np.arange(1, 6))),
        (
            np.arange(1, 6),
            np.arange(2, 7),
            (2, 5),
            np.full(
                (2, 5),
                np.arange(2, 7) * special.gamma(1 + 1 / np.arange(1, 6)),
            ),
        ),
    ],
)
def test_weibull_moment(alpha, beta, size, expected):
    with Model() as model:
        Weibull("x", alpha=alpha, beta=beta, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "n, p, size, expected",
    [
        (7, 0.7, None, 5),
        (7, 0.3, 5, np.full(5, 2)),
        (10, np.arange(1, 6) / 10, None, np.arange(1, 6)),
        (10, np.arange(1, 6) / 10, (2, 5), np.full((2, 5), np.arange(1, 6))),
    ],
)
def test_binomial_moment(n, p, size, expected):
    with Model() as model:
        Binomial("x", n=n, p=p, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, size, expected",
    [
        (2.7, None, 2),
        (2.3, 5, np.full(5, 2)),
        (np.arange(1, 5), None, np.arange(1, 5)),
        (np.arange(1, 5), (2, 4), np.full((2, 4), np.arange(1, 5))),
    ],
)
def test_poisson_moment(mu, size, expected):
    with Model() as model:
        Poisson("x", mu=mu, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "n, p, size, expected",
    [
        (10, 0.7, None, 4),
        (10, 0.7, 5, np.full(5, 4)),
        (np.full(3, 10), np.arange(1, 4) / 10, None, np.array([90, 40, 23])),
        (
            10,
            np.arange(1, 4) / 10,
            (2, 3),
            np.full((2, 3), np.array([90, 40, 23])),
        ),
    ],
)
def test_negative_binomial_moment(n, p, size, expected):
    with Model() as model:
        NegativeBinomial("x", n=n, p=p, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "c, size, expected",
    [
        (1, None, 1),
        (1, 5, np.full(5, 1)),
        (np.arange(1, 6), None, np.arange(1, 6)),
    ],
)
def test_constant_moment(c, size, expected):
    with Model() as model:
        Constant("x", c=c, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "psi, theta, size, expected",
    [
        (0.9, 3.0, None, 2),
        (0.8, 2.9, 5, np.full(5, 2)),
        (0.2, np.arange(1, 5) * 5, None, np.arange(1, 5)),
        (0.2, np.arange(1, 5) * 5, (2, 4), np.full((2, 4), np.arange(1, 5))),
    ],
)
def test_zero_inflated_poisson_moment(psi, theta, size, expected):
    with Model() as model:
        ZeroInflatedPoisson("x", psi=psi, theta=theta, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "psi, n, p, size, expected",
    [
        (0.2, 7, 0.7, None, 4),
        (0.2, 7, 0.3, 5, np.full(5, 2)),
        (0.6, 25, np.arange(1, 6) / 10, None, np.arange(1, 6)),
        (
            0.6,
            25,
            np.arange(1, 6) / 10,
            (2, 5),
            np.full((2, 5), np.arange(1, 6)),
        ),
    ],
)
def test_zero_inflated_binomial_moment(psi, n, p, size, expected):
    with Model() as model:
        ZeroInflatedBinomial("x", psi=psi, n=n, p=p, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, s, size, expected",
    [
        (1, 1, None, 1),
        (1, 1, 5, np.full(5, 1)),
        (2, np.arange(1, 6), None, np.full(5, 2)),
        (
            np.arange(1, 6),
            np.arange(1, 6),
            (2, 5),
            np.full((2, 5), np.arange(1, 6)),
        ),
    ],
)
def test_logistic_moment(mu, s, size, expected):
    with Model() as model:
        Logistic("x", mu=mu, s=s, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, nu, sigma, size, expected",
    [
        (1, 1, None, None, 2),
        (1, 1, np.ones((2, 5)), None, np.full([2, 5], 2)),
        (1, 1, None, 5, np.full(5, 2)),
        (1, np.arange(1, 6), None, None, np.arange(2, 7)),
        (1, np.arange(1, 6), None, (2, 5), np.full((2, 5), np.arange(2, 7))),
    ],
)
def test_exgaussian_moment(mu, nu, sigma, size, expected):
    with Model() as model:
        ExGaussian("x", mu=mu, sigma=sigma, nu=nu, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "p, size, expected",
    [
        (0.5, None, 2),
        (0.2, 5, 5 * np.ones(5)),
        (np.linspace(0.25, 1, 4), None, [4, 2, 1, 1]),
        (np.linspace(0.25, 1, 4), (2, 4), np.full((2, 4), [4, 2, 1, 1])),
    ],
)
def test_geometric_moment(p, size, expected):
    with Model() as model:
        Geometric("x", p=p, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "N, k, n, size, expected",
    [
        (50, 10, 20, None, 4),
        (50, 10, 23, 5, np.full(5, 5)),
        (50, 10, np.arange(23, 28), None, np.full(5, 5)),
        (
            50,
            10,
            np.arange(18, 23),
            (2, 5),
            np.full((2, 5), 4),
        ),
    ],
)
def test_hyper_geometric_moment(N, k, n, size, expected):
    with Model() as model:
        HyperGeometric("x", N=N, k=k, n=n, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "lower, upper, size, expected",
    [
        (1, 5, None, 3),
        (1, 5, 5, np.full(5, 3)),
        (1, np.arange(5, 22, 4), None, np.arange(3, 13, 2)),
        (
            1,
            np.arange(5, 22, 4),
            (2, 5),
            np.full((2, 5), np.arange(3, 13, 2)),
        ),
    ],
)
def test_discrete_uniform_moment(lower, upper, size, expected):
    with Model() as model:
        DiscreteUniform("x", lower=lower, upper=upper, size=size)


@pytest.mark.parametrize(
    "a, size, expected",
    [
        (
            np.array([2, 3, 5, 7, 11]),
            None,
            np.array([2, 3, 5, 7, 11]) / 28,
        ),
        (
            np.array([[1, 2, 3], [5, 6, 7]]),
            None,
            np.array([[1, 2, 3], [5, 6, 7]]) / np.array([6, 18])[..., np.newaxis],
        ),
        (
            np.array([[1, 2, 3], [5, 6, 7]]),
            7,
            np.apply_along_axis(
                lambda x: np.divide(x, np.array([6, 18])),
                1,
                np.broadcast_to([[1, 2, 3], [5, 6, 7]], shape=[7, 2, 3]),
            ),
        ),
        (
            np.full(shape=np.array([7, 3]), fill_value=np.array([13, 17, 19])),
            (
                11,
                5,
            ),
            np.broadcast_to([13, 17, 19], shape=[11, 5, 7, 3]) / 49,
        ),
    ],
)
def test_dirichlet_moment(a, size, expected):
    with Model() as model:
        Dirichlet("x", a=a, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, beta, size, expected",
    [
        (0, 2, None, 2 * np.euler_gamma),
        (1, np.arange(1, 4), None, 1 + np.arange(1, 4) * np.euler_gamma),
        (np.arange(5), 2, None, np.arange(5) + 2 * np.euler_gamma),
        (1, 2, 5, np.full(5, 1 + 2 * np.euler_gamma)),
        (
            np.arange(5),
            np.arange(1, 6),
            (2, 5),
            np.full((2, 5), np.arange(5) + np.arange(1, 6) * np.euler_gamma),
        ),
    ],
)
def test_gumbel_moment(mu, beta, size, expected):
    with Model() as model:
        Gumbel("x", mu=mu, beta=beta, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "c, lower, upper, size, expected",
    [
        (1, 0, 5, None, 2),
        (3, np.arange(-3, 6, 3), np.arange(3, 12, 3), None, np.array([1, 3, 5])),
        (np.arange(-3, 6, 3), -3, 3, None, np.array([-1, 0, 1])),
        (3, -3, 6, 5, np.full(5, 2)),
        (
            np.arange(-3, 6, 3),
            np.arange(-9, -2, 3),
            np.arange(3, 10, 3),
            (2, 3),
            np.full((2, 3), np.array([-3, 0, 3])),
        ),
    ],
)
def test_triangular_moment(c, lower, upper, size, expected):
    with Model() as model:
        Triangular("x", c=c, lower=lower, upper=upper, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, sigma, size, expected",
    [
        (1, 2, None, special.expit(1)),
        (0, np.arange(1, 5), None, special.expit(np.zeros(4))),
        (np.arange(4), 1, None, special.expit(np.arange(4))),
        (1, 5, 4, special.expit(np.ones(4))),
        (np.arange(4), np.arange(1, 5), (2, 4), np.full((2, 4), special.expit(np.arange(4)))),
    ],
)
def test_logitnormal_moment(mu, sigma, size, expected):
    with Model() as model:
        LogitNormal("x", mu=mu, sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "p, size, expected",
    [
        (np.array([0.1, 0.3, 0.6]), None, 2),
        (np.array([0.6, 0.1, 0.3]), 5, np.full(5, 0)),
        (np.full((2, 3), np.array([0.6, 0.1, 0.3])), None, [0, 0]),
        (
            np.full((2, 3), np.array([0.1, 0.3, 0.6])),
            (3, 2),
            np.full((3, 2), [2, 2]),
        ),
    ],
)
def test_categorical_moment(p, size, expected):
    with Model() as model:
        Categorical("x", p=p, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, cov, size, expected",
    [
        (np.ones(1), np.identity(1), None, np.ones(1)),
        (np.ones(3), np.identity(3), None, np.ones(3)),
        (np.ones((2, 2)), np.identity(2), None, np.ones((2, 2))),
        (np.array([1, 0, 3.0]), np.identity(3), None, np.array([1, 0, 3.0])),
        (np.array([1, 0, 3.0]), np.identity(3), (4, 2), np.full((4, 2, 3), [1, 0, 3.0])),
        (
            np.array([1, 3.0]),
            np.identity(2),
            5,
            np.full((5, 2), [1, 3.0]),
        ),
        (
            np.array([1, 3.0]),
            np.array([[1.0, 0.5], [0.5, 2]]),
            (4, 5),
            np.full((4, 5, 2), [1, 3.0]),
        ),
        (
            np.array([[3.0, 5], [1, 4]]),
            np.identity(2),
            (4, 5),
            np.full((4, 5, 2, 2), [[3.0, 5], [1, 4]]),
        ),
    ],
)
def test_mv_normal_moment(mu, cov, size, expected):
    with Model() as model:
        MvNormal("x", mu=mu, cov=cov, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, sigma, size, expected",
    [
        (4.0, 3.0, None, 7.8110885363844345),
        (4.0, np.full(5, 3), None, np.full(5, 7.8110885363844345)),
        (np.arange(5), 1, None, np.arange(5) + 1.2703628454614782),
        (np.arange(5), np.ones(5), (2, 5), np.full((2, 5), np.arange(5) + 1.2703628454614782)),
    ],
)
def test_moyal_moment(mu, sigma, size, expected):
    with Model() as model:
        Moyal("x", mu=mu, sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "alpha, mu, sigma, size, expected",
    [
        (1.0, 1.0, 1.0, None, 1.56418958),
        (1.0, np.ones(5), 1.0, None, np.full(5, 1.56418958)),
        (np.ones(5), 1, np.ones(5), None, np.full(5, 1.56418958)),
        (
            np.arange(5),
            np.arange(1, 6),
            np.arange(1, 6),
            None,
            (1.0, 3.12837917, 5.14094894, 7.02775903, 8.87030861),
        ),
        (
            np.arange(5),
            np.arange(1, 6),
            np.arange(1, 6),
            (2, 5),
            np.full((2, 5), (1.0, 3.12837917, 5.14094894, 7.02775903, 8.87030861)),
        ),
    ],
)
def test_skewnormal_moment(alpha, mu, sigma, size, expected):
    with Model() as model:
        SkewNormal("x", alpha=alpha, mu=mu, sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "b, kappa, mu, size, expected",
    [
        (1.0, 1.0, 1.0, None, 1.0),
        (1.0, np.ones(5), 1.0, None, np.full(5, 1.0)),
        (np.arange(1, 6), 1.0, np.ones(5), None, np.full(5, 1.0)),
        (
            np.arange(1, 6),
            np.arange(1, 6),
            np.arange(1, 6),
            None,
            (1.0, 1.25, 2.111111111111111, 3.0625, 4.04),
        ),
        (
            np.arange(1, 6),
            np.arange(1, 6),
            np.arange(1, 6),
            (2, 5),
            np.full((2, 5), (1.0, 1.25, 2.111111111111111, 3.0625, 4.04)),
        ),
    ],
)
def test_asymmetriclaplace_moment(b, kappa, mu, size, expected):
    with Model() as model:
        AsymmetricLaplace("x", b=b, kappa=kappa, mu=mu, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "nu, sigma, size, expected",
    [
        (1.0, 1.0, None, 1.5485724605511453),
        (1.0, np.ones(5), None, np.full(5, 1.5485724605511453)),
        (
            np.arange(1, 6),
            1.0,
            None,
            (
                1.5485724605511453,
                2.2723834280687427,
                3.1725772879007166,
                4.127193542536757,
                5.101069639492123,
            ),
        ),
        (
            np.arange(1, 6),
            np.ones(5),
            (2, 5),
            np.full(
                (2, 5),
                (
                    1.5485724605511453,
                    2.2723834280687427,
                    3.1725772879007166,
                    4.127193542536757,
                    5.101069639492123,
                ),
            ),
        ),
    ],
)
def test_rice_moment(nu, sigma, size, expected):
    with Model() as model:
        Rice("x", nu=nu, sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "get_moment, size, expected",
    [
        (None, None, 0.0),
        (None, 5, np.zeros(5)),
        ("custom_moment", None, 5),
        ("custom_moment", (2, 5), np.full((2, 5), 5)),
    ],
)
def test_density_dist_default_moment_univariate(get_moment, size, expected):
    if get_moment == "custom_moment":
        get_moment = lambda rv, size, *rv_inputs: 5 * at.ones(size, dtype=rv.dtype)
    with Model() as model:
        DensityDist("x", get_moment=get_moment, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize("size", [(), (2,), (3, 2)], ids=str)
def test_density_dist_custom_moment_univariate(size):
    def moment(rv, size, mu):
        return (at.ones(size) * mu).astype(rv.dtype)

    mu_val = np.array(np.random.normal(loc=2, scale=1)).astype(aesara.config.floatX)
    with pm.Model():
        mu = pm.Normal("mu")
        a = pm.DensityDist("a", mu, get_moment=moment, size=size)
    evaled_moment = get_moment(a).eval({mu: mu_val})
    assert evaled_moment.shape == to_tuple(size)
    assert np.all(evaled_moment == mu_val)


@pytest.mark.parametrize("size", [(), (2,), (3, 2)], ids=str)
def test_density_dist_custom_moment_multivariate(size):
    def moment(rv, size, mu):
        return (at.ones(size)[..., None] * mu).astype(rv.dtype)

    mu_val = np.random.normal(loc=2, scale=1, size=5).astype(aesara.config.floatX)
    with pm.Model():
        mu = pm.Normal("mu", size=5)
        a = pm.DensityDist("a", mu, get_moment=moment, ndims_params=[1], ndim_supp=1, size=size)
    evaled_moment = get_moment(a).eval({mu: mu_val})
    assert evaled_moment.shape == to_tuple(size) + (5,)
    assert np.all(evaled_moment == mu_val)


@pytest.mark.parametrize(
    "with_random, size",
    [
        (True, ()),
        (True, (2,)),
        (True, (3, 2)),
        (False, ()),
        (False, (2,)),
    ],
)
def test_density_dist_default_moment_multivariate(with_random, size):
    def _random(mu, rng=None, size=None):
        return rng.normal(mu, scale=1, size=to_tuple(size) + mu.shape)

    if with_random:
        random = _random
    else:
        random = None

    mu_val = np.random.normal(loc=2, scale=1, size=5).astype(aesara.config.floatX)
    with pm.Model():
        mu = pm.Normal("mu", size=5)
        a = pm.DensityDist("a", mu, random=random, ndims_params=[1], ndim_supp=1, size=size)
    if with_random:
        evaled_moment = get_moment(a).eval({mu: mu_val})
        assert evaled_moment.shape == to_tuple(size) + (5,)
        assert np.all(evaled_moment == 0)
    else:
        with pytest.raises(
            TypeError,
            match="Cannot safely infer the size of a multivariate random variable's moment.",
        ):
            evaled_moment = get_moment(a).eval({mu: mu_val})
