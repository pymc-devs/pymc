import aesara
import numpy as np
import pytest
import scipy.stats as st

from aesara import tensor as at
from scipy import special

import pymc as pm

from pymc.distributions import (
    CAR,
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
    DirichletMultinomial,
    DiscreteUniform,
    DiscreteWeibull,
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
    Interpolated,
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
    Simulator,
    SkewNormal,
    StickBreakingWeights,
    StudentT,
    Triangular,
    TruncatedNormal,
    Uniform,
    VonMises,
    Wald,
    Weibull,
    ZeroInflatedBinomial,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
from pymc.distributions.distribution import _moment, moment
from pymc.distributions.logprob import joint_logpt
from pymc.distributions.shape_utils import rv_size_is_none, to_tuple
from pymc.initial_point import make_initial_point_fn
from pymc.model import Model


def test_all_distributions_have_moments():
    import pymc.distributions as dist_module

    from pymc.distributions.distribution import DistributionMeta

    dists = (getattr(dist_module, dist) for dist in dist_module.__all__)
    dists = (dist for dist in dists if isinstance(dist, DistributionMeta))
    missing_moments = {
        dist for dist in dists if type(getattr(dist, "rv_op", None)) not in _moment.registry
    }

    # Ignore super classes
    missing_moments -= {
        dist_module.Distribution,
        dist_module.Discrete,
        dist_module.Continuous,
        dist_module.NoDistribution,
        dist_module.DensityDist,
        dist_module.simulator.Simulator,
    }

    # Distributions that have not been refactored for V4 yet
    not_implemented = {
        dist_module.timeseries.AR,
        dist_module.timeseries.AR1,
        dist_module.timeseries.GARCH11,
        dist_module.timeseries.GaussianRandomWalk,
        dist_module.timeseries.MvGaussianRandomWalk,
        dist_module.timeseries.MvStudentTRandomWalk,
    }

    # Distributions that have been refactored but don't yet have moments
    not_implemented |= {
        dist_module.multivariate.Wishart,
    }

    unexpected_implemented = not_implemented - missing_moments
    if unexpected_implemented:
        raise Exception(
            f"Distributions {unexpected_implemented} have a `moment` implemented. "
            "This test must be updated to expect this."
        )

    unexpected_not_implemented = missing_moments - not_implemented
    if unexpected_not_implemented:
        raise NotImplementedError(
            f"Unexpected by this test, distributions {unexpected_not_implemented} do "
            "not have a `moment` implementation. Either add a moment or filter "
            "these distributions in this test."
        )


def test_rv_size_is_none():
    rv = Normal.dist(0, 1, size=None)
    assert rv_size_is_none(rv.owner.inputs[1])

    rv = Normal.dist(0, 1, size=())
    assert rv_size_is_none(rv.owner.inputs[1])

    rv = Normal.dist(0, 1, size=1)
    assert not rv_size_is_none(rv.owner.inputs[1])

    size = Bernoulli.dist(0.5)
    rv = Normal.dist(0, 1, size=size)
    assert not rv_size_is_none(rv.owner.inputs[1])

    size = Normal.dist(0, 1).size
    rv = Normal.dist(0, 1, size=size)
    assert not rv_size_is_none(rv.owner.inputs[1])


def assert_moment_is_expected(model, expected, check_finite_logp=True):
    fn = make_initial_point_fn(
        model=model,
        return_transformed=False,
        default_strategy="moment",
    )
    moment = fn(0)["x"]
    expected = np.asarray(expected)
    try:
        random_draw = model["x"].eval()
    except NotImplementedError:
        random_draw = moment

    assert moment.shape == expected.shape
    assert expected.shape == random_draw.shape
    assert np.allclose(moment, expected)

    if check_finite_logp:
        logp_moment = joint_logpt(model["x"], at.constant(moment), transformed=False).eval()
        assert np.isfinite(logp_moment)


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
        (np.arange(1, 6), None, np.arange(1, 6)),
        (np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(1, 6))),
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
        (1, np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(1, 6))),
        (np.arange(1, 6), 1, None, np.full(5, 1)),
    ],
)
def test_halfstudentt_moment(nu, sigma, size, expected):
    with Model() as model:
        HalfStudentT("x", nu=nu, sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, sigma, lower, upper, size, expected",
    [
        (0.9, 1, -5, 5, None, 0),
        (1, np.ones(5), -10, np.inf, None, np.full(5, -9)),
        (np.arange(5), 1, None, 10, (2, 5), np.full((2, 5), 9)),
        (1, 1, [-np.inf, -np.inf, -np.inf], 10, None, np.full(3, 9)),
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
        (np.arange(1, 5), None, np.arange(1, 5)),
        (
            np.arange(1, 5),
            (2, 4),
            np.full((2, 4), np.arange(1, 5)),
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
    "alpha, beta, size, expected",
    [
        (5, 1, None, 1 / 4),
        (0.5, 1, None, 1 / 1.5),
        (5, 1, 5, np.full(5, 1 / (5 - 1))),
        (np.arange(1, 6), 1, None, np.array([0.5, 1, 1 / 2, 1 / 3, 1 / 4])),
    ],
)
def test_inverse_gamma_moment(alpha, beta, size, expected):
    with Model() as model:
        InverseGamma("x", alpha=alpha, beta=beta, size=size)
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
    "mu, kappa, size, expected",
    [
        (0, 1, None, 0),
        (0, np.ones(4), None, np.zeros(4)),
        (np.arange(4), 0.5, None, np.arange(4)),
        (np.arange(4), np.arange(1, 5), (2, 4), np.full((2, 4), np.arange(4))),
    ],
)
def test_vonmises_moment(mu, kappa, size, expected):
    with Model() as model:
        VonMises("x", mu=mu, kappa=kappa, size=size)
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
    "psi, mu, size, expected",
    [
        (0.9, 3.0, None, 3),
        (0.8, 2.9, 5, np.full(5, 2)),
        (0.2, np.arange(1, 5) * 5, None, np.arange(1, 5)),
        (0.2, np.arange(1, 5) * 5, (2, 4), np.full((2, 4), np.arange(1, 5))),
    ],
)
def test_zero_inflated_poisson_moment(psi, mu, size, expected):
    with Model() as model:
        ZeroInflatedPoisson("x", psi=psi, mu=mu, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "psi, n, p, size, expected",
    [
        (0.8, 7, 0.7, None, 4),
        (0.8, 7, 0.3, 5, np.full(5, 2)),
        (0.4, 25, np.arange(1, 6) / 10, None, np.arange(1, 6)),
        (
            0.4,
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
        (1, 1, 1, None, 2),
        (1, 1, np.ones((2, 5)), None, np.full([2, 5], 2)),
        (1, 1, 3, 5, np.full(5, 2)),
        (1, np.arange(1, 6), 5, None, np.arange(2, 7)),
        (1, np.arange(1, 6), 1, (2, 5), np.full((2, 5), np.arange(2, 7))),
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
        assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "q, beta, size, expected",
    [
        (0.5, 0.5, None, 0),
        (0.6, 0.1, 5, (20,) * 5),
        (np.linspace(0.25, 0.99, 4), 0.42, None, [0, 0, 6, 23862]),
        (
            np.linspace(0.5, 0.99, 3),
            [[1, 1.25, 1.75], [1.25, 0.75, 0.5]],
            None,
            [[0, 0, 10], [0, 2, 4755]],
        ),
    ],
)
def test_discrete_weibull_moment(q, beta, size, expected):
    with Model() as model:
        DiscreteWeibull("x", q=q, beta=beta, size=size)
    assert_moment_is_expected(model, expected)


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
            (7, 2),
            np.apply_along_axis(
                lambda x: np.divide(x, np.array([6, 18])),
                1,
                np.broadcast_to([[1, 2, 3], [5, 6, 7]], shape=[7, 2, 3]),
            ),
        ),
        (
            np.full(shape=np.array([7, 3]), fill_value=np.array([13, 17, 19])),
            (11, 5, 7),
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
    "x_points, pdf_points, size, expected",
    [
        (np.array([-1, 1]), np.array([0.4, 0.6]), None, 0.2),
        (
            np.array([-4, -1, 3, 9, 19]),
            np.array([0.1, 0.15, 0.2, 0.25, 0.3]),
            None,
            1.5458937198067635,
        ),
        (
            np.array([-22, -4, 0, 8, 13]),
            np.tile(1 / 5, 5),
            (5, 3),
            np.full((5, 3), -0.14285714285714296),
        ),
        (
            np.arange(-100, 10),
            np.arange(1, 111) / 6105,
            (2, 5, 3),
            np.full((2, 5, 3), -27.584097859327223),
        ),
    ],
)
def test_interpolated_moment(x_points, pdf_points, size, expected):
    with Model() as model:
        Interpolated("x", x_points=x_points, pdf_points=pdf_points, size=size)
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
            (4, 5, 2),
            np.full((4, 5, 2, 2), [[3.0, 5], [1, 4]]),
        ),
    ],
)
def test_mv_normal_moment(mu, cov, size, expected):
    with Model() as model:
        x = MvNormal("x", mu=mu, cov=cov, size=size)

    # MvNormal logp is only implemented for up to 2D variables
    assert_moment_is_expected(model, expected, check_finite_logp=x.ndim < 3)


@pytest.mark.parametrize(
    "mu, size, expected",
    [
        (
            np.array([1, 0, 3.0, 4]),
            None,
            np.array([1, 0, 3.0, 4]),
        ),
        (np.array([1, 0, 3.0, 4]), 6, np.full((6, 4), [1, 0, 3.0, 4])),
        (np.array([1, 0, 3.0, 4]), (5, 3), np.full((5, 3, 4), [1, 0, 3.0, 4])),
        (
            np.array([[3.0, 5, 2, 1], [1, 4, 0.5, 9]]),
            (4, 5, 2),
            np.full((4, 5, 2, 4), [[3.0, 5, 2, 1], [1, 4, 0.5, 9]]),
        ),
    ],
)
def test_car_moment(mu, size, expected):
    W = np.array(
        [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
    )
    tau = 2
    alpha = 0.5
    with Model() as model:
        CAR("x", mu=mu, W=W, alpha=alpha, tau=tau, size=size)
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


rand1d = np.random.rand(2)
rand2d = np.random.rand(2, 3)


@pytest.mark.parametrize(
    "nu, mu, cov, size, expected",
    [
        (2, np.ones(1), np.eye(1), None, np.ones(1)),
        (2, rand1d, np.eye(2), None, rand1d),
        (2, rand1d, np.eye(2), 2, np.full((2, 2), rand1d)),
        (2, rand1d, np.eye(2), (2, 5), np.full((2, 5, 2), rand1d)),
        (2, rand2d, np.eye(3), None, rand2d),
        (2, rand2d, np.eye(3), (2, 2), np.full((2, 2, 3), rand2d)),
        (2, rand2d, np.eye(3), (2, 5, 2), np.full((2, 5, 2, 3), rand2d)),
    ],
)
def test_mvstudentt_moment(nu, mu, cov, size, expected):
    with Model() as model:
        x = MvStudentT("x", nu=nu, mu=mu, cov=cov, size=size)

    # MvStudentT logp is only impemented for up to 2D variables
    assert_moment_is_expected(model, expected, check_finite_logp=x.ndim < 3)


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
    "mu, rowchol, colchol, size, expected",
    [
        (np.ones((1, 1)), np.eye(1), np.eye(1), None, np.ones((1, 1))),
        (np.ones((1, 1)), np.eye(2), np.eye(3), None, np.ones((2, 3))),
        (rand2d, np.eye(2), np.eye(3), None, rand2d),
        (rand2d, np.eye(2), np.eye(3), 2, np.full((2, 2, 3), rand2d)),
        (rand2d, np.eye(2), np.eye(3), (2, 5), np.full((2, 5, 2, 3), rand2d)),
    ],
)
def test_matrixnormal_moment(mu, rowchol, colchol, size, expected):
    with Model() as model:
        x = MatrixNormal("x", mu=mu, rowchol=rowchol, colchol=colchol, size=size)

    # MatrixNormal logp is only implemented for 2d values
    check_logp = x.ndim == 2
    assert_moment_is_expected(model, expected, check_finite_logp=check_logp)


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


@pytest.mark.parametrize(
    "alpha, K, size, expected",
    [
        (3, 11, None, np.append((3 / 4) ** np.arange(11) * 1 / 4, (3 / 4) ** 11)),
        (5, 19, None, np.append((5 / 6) ** np.arange(19) * 1 / 6, (5 / 6) ** 19)),
        (
            1,
            7,
            (13,),
            np.full(
                shape=(13, 8), fill_value=np.append((1 / 2) ** np.arange(7) * 1 / 2, (1 / 2) ** 7)
            ),
        ),
        (
            0.5,
            5,
            (3, 5, 7),
            np.full(
                shape=(3, 5, 7, 6),
                fill_value=np.append((1 / 3) ** np.arange(5) * 2 / 3, (1 / 3) ** 5),
            ),
        ),
    ],
)
def test_stickbreakingweights_moment(alpha, K, size, expected):
    with Model() as model:
        StickBreakingWeights("x", alpha=alpha, K=K, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "moment, size, expected",
    [
        (None, None, 0.0),
        (None, 5, np.zeros(5)),
        ("custom_moment", None, 5),
        ("custom_moment", (2, 5), np.full((2, 5), 5)),
    ],
)
def test_density_dist_default_moment_univariate(moment, size, expected):
    if moment == "custom_moment":
        moment = lambda rv, size, *rv_inputs: 5 * at.ones(size, dtype=rv.dtype)
    with Model() as model:
        DensityDist("x", moment=moment, size=size)
    assert_moment_is_expected(model, expected, check_finite_logp=False)


@pytest.mark.parametrize("size", [(), (2,), (3, 2)], ids=str)
def test_density_dist_custom_moment_univariate(size):
    def density_moment(rv, size, mu):
        return (at.ones(size) * mu).astype(rv.dtype)

    mu_val = np.array(np.random.normal(loc=2, scale=1)).astype(aesara.config.floatX)
    with pm.Model():
        mu = pm.Normal("mu")
        a = pm.DensityDist("a", mu, moment=density_moment, size=size)
    evaled_moment = moment(a).eval({mu: mu_val})
    assert evaled_moment.shape == to_tuple(size)
    assert np.all(evaled_moment == mu_val)


@pytest.mark.parametrize("size", [(), (2,), (3, 2)], ids=str)
def test_density_dist_custom_moment_multivariate(size):
    def density_moment(rv, size, mu):
        return (at.ones(size)[..., None] * mu).astype(rv.dtype)

    mu_val = np.random.normal(loc=2, scale=1, size=5).astype(aesara.config.floatX)
    with pm.Model():
        mu = pm.Normal("mu", size=5)
        a = pm.DensityDist("a", mu, moment=density_moment, ndims_params=[1], ndim_supp=1, size=size)
    evaled_moment = moment(a).eval({mu: mu_val})
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
        evaled_moment = moment(a).eval({mu: mu_val})
        assert evaled_moment.shape == to_tuple(size) + (5,)
        assert np.all(evaled_moment == 0)
    else:
        with pytest.raises(
            TypeError,
            match="Cannot safely infer the size of a multivariate random variable's moment.",
        ):
            evaled_moment = moment(a).eval({mu: mu_val})


@pytest.mark.parametrize(
    "h, z, size, expected",
    [
        (1.0, 0.0, None, 0.25),
        (
            1.0,
            np.arange(5),
            None,
            (
                0.25,
                0.23105857863000487,
                0.1903985389889412,
                0.1508580422741444,
                0.12050344750947711,
            ),
        ),
        (
            np.arange(1, 6),
            np.arange(5),
            None,
            (
                0.25,
                0.46211715726000974,
                0.5711956169668236,
                0.6034321690965776,
                0.6025172375473855,
            ),
        ),
        (
            np.arange(1, 6),
            np.arange(5),
            (2, 5),
            np.full(
                (2, 5),
                (
                    0.25,
                    0.46211715726000974,
                    0.5711956169668236,
                    0.6034321690965776,
                    0.6025172375473855,
                ),
            ),
        ),
    ],
)
def test_polyagamma_moment(h, z, size, expected):
    with Model() as model:
        PolyaGamma("x", h=h, z=z, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "p, n, size, expected",
    [
        (np.array([0.25, 0.25, 0.25, 0.25]), 1, None, np.array([1, 0, 0, 0])),
        (np.array([0.3, 0.6, 0.05, 0.05]), 2, None, np.array([1, 1, 0, 0])),
        (np.array([0.3, 0.6, 0.05, 0.05]), 10, None, np.array([4, 6, 0, 0])),
        (
            np.array([[0.3, 0.6, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25]]),
            10,
            None,
            np.array([[4, 6, 0, 0], [4, 2, 2, 2]]),
        ),
        (
            np.array([0.3, 0.6, 0.05, 0.05]),
            np.array([2, 10]),
            (1, 2),
            np.array([[[1, 1, 0, 0], [4, 6, 0, 0]]]),
        ),
        (
            np.array([[0.25, 0.25, 0.25, 0.25], [0.26, 0.26, 0.26, 0.22]]),
            np.array([1, 10]),
            None,
            np.array([[1, 0, 0, 0], [2, 3, 3, 2]]),
        ),
        (
            np.array([[0.25, 0.25, 0.25, 0.25], [0.26, 0.26, 0.26, 0.22]]),
            np.array([1, 10]),
            (3, 2),
            np.full((3, 2, 4), [[1, 0, 0, 0], [2, 3, 3, 2]]),
        ),
    ],
)
def test_multinomial_moment(p, n, size, expected):
    with Model() as model:
        Multinomial("x", n=n, p=p, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "psi, mu, alpha, size, expected",
    [
        (0.2, 10, 3, None, 2),
        (0.2, 10, 4, 5, np.full(5, 2)),
        (
            0.4,
            np.arange(1, 5),
            np.arange(2, 6),
            None,
            np.array([0, 1, 1, 2] if aesara.config.floatX == "float64" else [0, 0, 1, 1]),
        ),
        (
            np.linspace(0.2, 0.6, 3),
            np.arange(1, 10, 4),
            np.arange(1, 4),
            (2, 3),
            np.full((2, 3), np.array([0, 2, 5])),
        ),
    ],
)
def test_zero_inflated_negative_binomial_moment(psi, mu, alpha, size, expected):
    with Model() as model:
        ZeroInflatedNegativeBinomial("x", psi=psi, mu=mu, alpha=alpha, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize("mu", [0, np.arange(3)], ids=str)
@pytest.mark.parametrize("sigma", [1, np.array([1, 2, 5])], ids=str)
@pytest.mark.parametrize("size", [None, 3, (5, 3)], ids=str)
def test_simulator_moment(mu, sigma, size):
    def normal_sim(rng, mu, sigma, size):
        return rng.normal(mu, sigma, size=size)

    with Model() as model:
        x = Simulator("x", normal_sim, mu, sigma, size=size)

    fn = make_initial_point_fn(
        model=model,
        return_transformed=False,
        default_strategy="moment",
    )

    random_draw = model["x"].eval()
    result = fn(0)["x"]
    assert result.shape == random_draw.shape

    # We perform a z-test between the moment and expected mean from a sample of 10 draws
    # This test fails if the number of samples averaged in moment(Simulator)
    # is much smaller than 10, but would not catch the case where the number of samples
    # is higher than the expected 10

    n = 10  # samples
    expected_sample_mean = mu
    expected_sample_mean_std = np.sqrt(sigma**2 / n)

    # Multiple test adjustment for z-test to maintain alpha=0.01
    alpha = 0.01
    alpha /= 2 * 2 * 3  # Correct for number of test permutations
    alpha /= random_draw.size  # Correct for distribution size
    cutoff = st.norm().ppf(1 - (alpha / 2))

    assert np.all(np.abs((result - expected_sample_mean) / expected_sample_mean_std) < cutoff)


@pytest.mark.parametrize(
    "mu, covs, size, expected",
    [
        (np.ones(1), [np.identity(1), np.identity(1)], None, np.ones(1)),
        (np.ones(6), [np.identity(2), np.identity(3)], 5, np.ones((5, 6))),
        (np.zeros(6), [np.identity(2), np.identity(3)], 6, np.zeros((6, 6))),
        (np.zeros(3), [np.identity(3), np.identity(1)], 6, np.zeros((6, 3))),
        (
            np.array([1, 2, 3, 4]),
            [
                np.array([[1.0, 0.5], [0.5, 2]]),
                np.array([[1.0, 0.4], [0.4, 2]]),
            ],
            2,
            np.array(
                [
                    [1, 2, 3, 4],
                    [1, 2, 3, 4],
                ]
            ),
        ),
    ],
)
def test_kronecker_normal_moment(mu, covs, size, expected):
    with Model() as model:
        KroneckerNormal("x", mu=mu, covs=covs, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "n, eta, size, expected",
    [
        (3, 1, None, np.zeros(3)),
        (5, 1, None, np.zeros(10)),
        (3, 1, 1, np.zeros((1, 3))),
        (5, 1, (2, 3), np.zeros((2, 3, 10))),
    ],
)
def test_lkjcorr_moment(n, eta, size, expected):
    with Model() as model:
        LKJCorr("x", n=n, eta=eta, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "n, eta, size, expected",
    [
        (3, 1, None, np.array([1, 0, 1, 0, 0, 1])),
        (4, 1, None, np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])),
        (3, 1, 1, np.array([[1, 0, 1, 0, 0, 1]])),
        (
            4,
            1,
            (2, 3),
            np.full((2, 3, 10), np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])),
        ),
    ],
)
def test_lkjcholeskycov_moment(n, eta, size, expected):
    with Model() as model:
        sd_dist = pm.Exponential.dist(1, size=(*to_tuple(size), n))
        LKJCholeskyCov("x", n=n, eta=eta, sd_dist=sd_dist, size=size, compute_corr=False)
    assert_moment_is_expected(model, expected, check_finite_logp=size is None)


@pytest.mark.parametrize(
    "a, n, size, expected",
    [
        (np.array([2, 2, 2, 2]), 1, None, np.array([1, 0, 0, 0])),
        (np.array([3, 6, 0.5, 0.5]), 2, None, np.array([1, 1, 0, 0])),
        (np.array([30, 60, 5, 5]), 10, None, np.array([4, 6, 0, 0])),
        (
            np.array([[30, 60, 5, 5], [26, 26, 26, 22]]),
            10,
            (1, 2),
            np.array([[[4, 6, 0, 0], [2, 3, 3, 2]]]),
        ),
        (
            np.array([26, 26, 26, 22]),
            np.array([1, 10]),
            None,
            np.array([[1, 0, 0, 0], [2, 3, 3, 2]]),
        ),
        (
            np.array([[26, 26, 26, 22]]),  # Dim: 1 x 4
            np.array([[1], [10]]),  # Dim: 2 x 1
            (2, 1, 2, 1),
            np.full(
                (2, 1, 2, 1, 4),
                np.array([[[1, 0, 0, 0]], [[2, 3, 3, 2]]]),  # Dim: 2 x 1 x 4
            ),
        ),
    ],
)
def test_dirichlet_multinomial_moment(a, n, size, expected):
    with Model() as model:
        DirichletMultinomial("x", n=n, a=a, size=size)
    assert_moment_is_expected(model, expected)
