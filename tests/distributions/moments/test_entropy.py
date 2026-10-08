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

import numpy as np
import pytest

from pytensor import function
from pytensor import tensor as pt
from pytensor.compile.mode import Mode
from scipy import stats
from scipy.stats import (
    bernoulli,
    beta,
    cauchy,
    chi2,
    dirichlet,
    expon,
    gamma,
    geom,
    gumbel_r,
    halfcauchy,
    halfnorm,
    invgamma,
    laplace,
    laplace_asymmetric,
    logistic,
    lognorm,
    moyal,
    multivariate_normal,
    norm,
    pareto,
    randint,
    t,
    triang,
    uniform,
    vonmises,
    weibull_min,
)

from pymc import (
    AsymmetricLaplace,
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Cauchy,
    ChiSquared,
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
    InverseGamma,
    Kumaraswamy,
    Laplace,
    Logistic,
    LogitNormal,
    LogNormal,
    Moyal,
    MvNormal,
    MvStudentT,
    Normal,
    Pareto,
    Poisson,
    Rice,
    SkewNormal,
    StudentT,
    Triangular,
    Uniform,
    VonMises,
    Wald,
    Weibull,
)
from pymc.distributions.moments.entropy import entropy

mode = Mode(linker="py", optimizer=None)


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
        [Cauchy, cauchy, {"alpha": 2, "beta": 1.5}, {"loc": 2, "scale": 1.5}],
        [ChiSquared, chi2, {"nu": 6}, {"df": 6}],
        [DiscreteUniform, randint, {"lower": 2, "upper": 8}, {"low": 2, "high": 9}],
        [Exponential, expon, {"lam": 0.5}, {"scale": 2}],
        [Gamma, gamma, {"alpha": 4, "beta": 3}, {"a": 4, "scale": 1 / 3}],
        [Geometric, geom, {"p": 0.1}, {"p": 0.1}],
        [Gumbel, gumbel_r, {"mu": 2, "beta": 1}, {"loc": 2, "scale": 1}],
        [HalfCauchy, halfcauchy, {"beta": 1.5}, {"scale": 1.5}],
        [HalfNormal, halfnorm, {"sigma": 3}, {"scale": 3}],
        [InverseGamma, invgamma, {"alpha": 2, "beta": 2}, {"a": 2, "scale": 2}],
        [Laplace, laplace, {"mu": 2, "b": 2}, {"loc": 2, "scale": 2}],
        [Logistic, logistic, {"mu": 2, "s": 1.5}, {"loc": 2, "scale": 1.5}],
        [LogNormal, lognorm, {"mu": 0.3, "sigma": 0.6}, {"scale": np.exp(0.3), "s": 0.6}],
        [Moyal, moyal, {"mu": 2, "sigma": 2}, {"loc": 2, "scale": 2}],
        [Normal, norm, {"mu": 2, "sigma": 3}, {"loc": 2, "scale": 3}],
        [Pareto, pareto, {"alpha": 5, "m": 2}, {"b": 5, "scale": 2}],
        [StudentT, t, {"nu": 6, "mu": 0, "sigma": 2}, {"df": 6, "loc": 0, "scale": 2}],
        [Triangular, triang, {"lower": 0, "c": 0.5, "upper": 2}, {"c": 0.25, "loc": 0, "scale": 2}],
        [Uniform, uniform, {"lower": -1, "upper": 4}, {"loc": -1, "scale": 5}],
        [VonMises, vonmises, {"mu": 1, "kappa": 2.5}, {"loc": 1, "kappa": 2.5}],
        [Weibull, weibull_min, {"alpha": 1.5, "beta": 2}, {"c": 1.5, "scale": 2}],
    ],
)
def test_entropy_equal_to_scipy(dist, scipy_equiv, dist_params, scipy_params):
    rv = dist.dist(**dist_params)
    pymc_entropy = entropy(rv).eval(mode=mode)
    scipy_entropy = scipy_equiv(**scipy_params).entropy()
    assert np.asarray(pymc_entropy).shape == np.asarray(scipy_entropy).shape
    np.testing.assert_allclose(pymc_entropy, scipy_entropy, rtol=1e-6)

    # entropy of a batched distribution broadcasts over the batch dimension
    pymc_entropy_tiled = entropy(dist.dist(shape=(3,), **dist_params)).eval()
    np.testing.assert_allclose(pymc_entropy_tiled, np.tile(pymc_entropy, 3), rtol=1e-6)


@pytest.mark.parametrize(
    ["dist", "dist_params", "expected"],
    [
        [Categorical, {"p": [0.1, 0.2, 0.3, 0.4]}, stats.entropy([0.1, 0.2, 0.3, 0.4])],
    ],
)
def test_entropy_equal_expected(dist, dist_params, expected):
    rv = dist.dist(**dist_params)
    pymc_entropy = entropy(rv).eval(mode=mode)
    np.testing.assert_allclose(pymc_entropy, expected, rtol=1e-6)


@pytest.mark.parametrize(
    ["dist", "dist_params", "scipy_entropy"],
    [
        [
            MvNormal,
            {
                "mu": np.zeros(3),
                "cov": np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]]),
            },
            multivariate_normal(
                np.zeros(3), np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
            ).entropy(),
        ],
        [
            Dirichlet,
            {"a": np.array([2.0, 3.0, 4.0, 1.0])},
            dirichlet(np.array([2.0, 3.0, 4.0, 1.0])).entropy(),
        ],
    ],
)
def test_entropy_multivariate(dist, dist_params, scipy_entropy):
    rv = dist.dist(**dist_params)
    pymc_entropy = entropy(rv).eval(mode=mode)
    assert np.asarray(pymc_entropy).shape == ()
    np.testing.assert_allclose(pymc_entropy, scipy_entropy, rtol=1e-6)

    # a batch of independent multivariate distributions yields one entropy per batch element
    pymc_entropy_batched = entropy(dist.dist(size=(4,), **dist_params)).eval()
    np.testing.assert_allclose(pymc_entropy_batched, np.full(4, pymc_entropy), rtol=1e-6)


@pytest.mark.parametrize(
    ["dist", "dist_params"],
    [
        [Binomial, {"n": 5, "p": 0.6}],
        [ExGaussian, {"mu": 0, "sigma": 1, "nu": 1}],
        [Flat, {}],
        [HalfFlat, {}],
        [HalfStudentT, {"nu": 3, "sigma": 1}],
        [Kumaraswamy, {"a": 2, "b": 2}],
        [LogitNormal, {"mu": 0, "sigma": 1}],
        [MvStudentT, {"mu": np.zeros(3), "scale": np.eye(3), "nu": 4}],
        [Poisson, {"mu": 3}],
        [Rice, {"nu": 1, "sigma": 1}],
        [SkewNormal, {"mu": 0, "sigma": 1, "alpha": 2}],
        [Wald, {"mu": 1, "lam": 1}],
    ],
)
def test_no_entropy(dist, dist_params):
    with pytest.raises(NotImplementedError):
        entropy(dist.dist(**dist_params))


@pytest.mark.parametrize(
    ["dist", "param", "value"],
    [
        # A regularization-by-entropy term (e.g. in RL policies) requires the
        # entropy to be differentiable w.r.t. the distribution parameters.
        [lambda p: Normal.dist(mu=0.0, sigma=p), "sigma", 2.0],
        [lambda p: Gamma.dist(alpha=p, beta=1.0), "alpha", 3.0],
        [lambda p: Beta.dist(alpha=p, beta=2.0), "alpha", 3.0],
        [lambda p: StudentT.dist(nu=p, mu=0.0, sigma=1.0), "nu", 5.0],
        [lambda p: VonMises.dist(mu=0.0, kappa=p), "kappa", 2.5],
    ],
)
def test_entropy_is_differentiable(dist, param, value):
    p = pt.scalar(param)
    grad = pt.grad(entropy(dist(p)).sum(), p)
    grad_fn = function([p], grad, mode=mode, on_unused_input="ignore")
    analytic = grad_fn(value)
    h = 1e-5
    entropy_fn = function([p], entropy(dist(p)), mode=mode, on_unused_input="ignore")
    finite_diff = (entropy_fn(value + h) - entropy_fn(value - h)) / (2 * h)
    np.testing.assert_allclose(analytic, finite_diff, rtol=1e-4)
