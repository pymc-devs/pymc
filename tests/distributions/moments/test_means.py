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

from pytensor.compile.mode import Mode
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
    matrix_normal,
    moyal,
    multinomial,
    multivariate_normal,
    multivariate_t,
    nbinom,
    norm,
    pareto,
    poisson,
    rice,
    skewnorm,
    t,
    triang,
    uniform,
    vonmises,
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
from pymc.distributions.moments.means import mean
from pymc.exceptions import UndefinedMomentException


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
        [InverseGamma, invgamma, {"alpha": 2, "beta": 2}, {"a": 2, "scale": 2}],
        [Laplace, laplace, {"mu": 2, "b": 2}, {"loc": 2, "scale": 2}],
        [Logistic, logistic, {"mu": 2, "s": 1}, {"loc": 2, "scale": 1}],
        [LogNormal, lognorm, {"mu": 0.3, "sigma": 0.6}, {"scale": np.exp(0.3), "s": 0.6}],
        [
            MatrixNormal,
            matrix_normal,
            {"mu": np.eye(3), "rowcov": np.eye(3), "colcov": np.eye(3)},
            {"mean": np.eye(3), "rowcov": np.eye(3), "colcov": np.eye(3)},
        ],
        [Moyal, moyal, {"mu": 2, "sigma": 2}, {"loc": 2, "scale": 2}],
        [Multinomial, multinomial, {"n": 20, "p": np.ones(6) / 6}, {"n": 20, "p": np.ones(6) / 6}],
        [
            MvNormal,
            multivariate_normal,
            {"mu": np.ones(3), "cov": np.eye(3)},
            {"mean": np.ones(3), "cov": np.eye(3)},
        ],
        [
            MvStudentT,
            multivariate_t,
            {"mu": np.ones(3), "cov": np.eye(3), "nu": 4},
            {"loc": np.ones(3), "shape": np.eye(3), "df": 4},
        ],
        [NegativeBinomial, nbinom, {"n": 10, "p": 0.5}, {"n": 10, "p": 0.5}],
        [Normal, norm, {"mu": 2, "sigma": 2}, {"loc": 2, "scale": 2}],
        [Pareto, pareto, {"alpha": 5, "m": 2}, {"b": 5, "scale": 2}],
        [Poisson, poisson, {"mu": 20}, {"mu": 20}],
        [Rice, rice, {"b": 2, "sigma": 2}, {"b": 2, "scale": 2}],
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
        [VonMises, vonmises, {"mu": 2, "kappa": 2}, {"loc": 2, "kappa": 2}],
        [Wald, invgauss, {"mu": 2, "lam": 1}, {"mu": 2, "scale": 1}],
        [Weibull, weibull_min, {"alpha": 2, "beta": 2}, {"c": 2, "scale": 2}],
    ],
)
def test_mean_equal_to_scipy(dist, scipy_equiv, dist_params, scipy_params):
    rv = dist.dist(**dist_params)
    mode = Mode(linker="py", optimizer=None)
    pymc_mean = mean(rv).eval(mode=mode)
    scipy_rv = scipy_equiv(**scipy_params)
    try:
        scipy_mean = scipy_rv.mean()
    except TypeError:
        # Happens for multivariate_normal
        scipy_mean = scipy_rv.mean
    except AttributeError:
        # Happens for multivariate_t
        scipy_mean = scipy_rv.loc
    assert np.asarray(pymc_mean).shape == np.asarray(scipy_mean).shape
    np.testing.assert_almost_equal(pymc_mean, scipy_mean)
    pymc_mean_tiled = mean(dist.dist(shape=(3, *pymc_mean.shape), **dist_params)).eval()
    np.testing.assert_almost_equal(
        pymc_mean_tiled, np.tile(pymc_mean, (3,) + (1,) * pymc_mean.ndim)
    )


@pytest.mark.parametrize(
    ["dist", "dist_params", "expected"],
    [
        [CAR, {"mu": np.ones(3), "W": np.eye(3), "alpha": 0.5, "tau": 1}, np.ones(3)],
        [DiracDelta, {"c": 4.0}, 4.0],
        [DirichletMultinomial, {"n": 5, "a": np.ones(5)}, np.ones(5)],
        [DiscreteUniform, {"lower": 3, "upper": 5}, 4.0],
        [HalfStudentT, {"nu": 2, "sigma": np.sqrt(2)}, 2.0],
        [
            KroneckerNormal,
            {
                "mu": np.ones(6),
                "covs": [
                    np.array([[1.0, 0.5], [0.5, 2]]),
                    np.array([[1.0, 0.4, 0.2], [0.4, 2, 0.3], [0.2, 0.3, 1]]),
                ],
            },
            np.ones(6),
        ],
        [Kumaraswamy, {"a": 1, "b": 1}, 0.5],
        [
            LKJCholeskyCov,
            {"eta": 1, "n": 3, "sd_dist": DiracDelta.dist(1), "compute_corr": False},
            np.eye(3)[np.tril_indices(3)],
        ],
        [LKJCorr, {"eta": 1, "n": 3}, np.eye(3)],
        [Mixture, {"w": [0.3, 0.7], "comp_dists": Normal.dist(mu=[0, 1], sigma=1)}, 0.7],
        [PolyaGamma, {"h": 1, "z": 1}, 0.23105858],
        [
            StickBreakingWeights,
            {"alpha": 1, "K": 5},
            np.concatenate([0.5 ** np.arange(1, 6), [0.5**5]]),
        ],
        [ZeroInflatedBinomial, {"n": 10, "p": 0.5, "psi": 0.8}, 4.0],
        [ZeroInflatedNegativeBinomial, {"n": 10, "p": 0.5, "psi": 0.8}, 8.0],
        [ZeroInflatedPoisson, {"mu": 5, "psi": 0.8}, 4.0],
    ],
)
def test_mean_equal_expected(dist, dist_params, expected):
    expected = np.asarray(expected)
    rv = dist.dist(**dist_params)
    mode = Mode(linker="py", optimizer=None)
    pymc_mean = mean(rv).eval(mode=mode)
    np.testing.assert_almost_equal(pymc_mean, expected)
    pymc_mean_tiled = mean(dist.dist(shape=(3, *pymc_mean.shape), **dist_params)).eval()
    np.testing.assert_almost_equal(
        pymc_mean_tiled, np.tile(pymc_mean, (3,) + (1,) * pymc_mean.ndim)
    )


def test_wishart_mean():
    nu = 10
    V = np.array([[2.0, 0.5], [0.5, 1.5]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pymc import Wishart

        rv = Wishart.dist(nu=nu, V=V)

    pymc_mean_val = mean(rv).eval()
    scipy_mean = wishart(df=nu, scale=V).mean()

    np.testing.assert_almost_equal(pymc_mean_val, scipy_mean)


@pytest.mark.parametrize(
    ["dist", "dist_params"],
    [
        [Cauchy, {"alpha": 1, "beta": 1}],
        [HalfCauchy, {"beta": 1.0}],
        [LogitNormal, {"mu": 2, "sigma": 1}],
        [Flat, {}],
        [HalfFlat, {}],
        [Categorical, {"p": [0.1, 0.9]}],
    ],
)
def test_no_mean(dist, dist_params):
    with pytest.raises((UndefinedMomentException, NotImplementedError)):
        mean(dist.dist(**dist_params))
