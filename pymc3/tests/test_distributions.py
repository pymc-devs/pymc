#   Copyright 2020 The PyMC Developers
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

import itertools
import sys

import numpy as np
import numpy.random as nr
import pytest
import scipy.stats
import scipy.stats.distributions as sp
import theano
import theano.tensor as tt

from numpy import array, exp, inf, log
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from packaging.version import parse
from scipy import __version__ as scipy_version
from scipy import integrate
from scipy.special import erf, logit

import pymc3 as pm

from pymc3.blocking import DictToVarBijection
from pymc3.distributions import (
    AR1,
    AsymmetricLaplace,
    Bernoulli,
    Beta,
    BetaBinomial,
    Binomial,
    Bound,
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
    LKJCorr,
    Logistic,
    LogitNormal,
    Lognormal,
    MatrixNormal,
    Moyal,
    Multinomial,
    MvNormal,
    MvStudentT,
    NegativeBinomial,
    Normal,
    OrderedLogistic,
    OrderedProbit,
    Pareto,
    Poisson,
    Rice,
    SkewNormal,
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
    continuous,
)
from pymc3.math import kronecker, logsumexp
from pymc3.model import Deterministic, Model, Point
from pymc3.tests.helpers import select_by_precision
from pymc3.theanof import floatX
from pymc3.vartypes import continuous_types

SCIPY_VERSION = parse(scipy_version)


def get_lkj_cases():
    """
    Log probabilities calculated using the formulas in:
    http://www.sciencedirect.com/science/article/pii/S0047259X09000876
    """
    tri = np.array([0.7, 0.0, -0.7])
    return [
        (tri, 1, 3, 1.5963125911388549),
        (tri, 3, 3, -7.7963493376312742),
        (tri, 0, 3, -np.inf),
        (np.array([1.1, 0.0, -0.7]), 1, 3, -np.inf),
        (np.array([0.7, 0.0, -1.1]), 1, 3, -np.inf),
    ]


LKJ_CASES = get_lkj_cases()


class Domain:
    def __init__(self, vals, dtype=None, edges=None, shape=None):
        avals = array(vals, dtype=dtype)
        if dtype is None and not str(avals.dtype).startswith("int"):
            avals = avals.astype(theano.config.floatX)
        vals = [array(v, dtype=avals.dtype) for v in vals]

        if edges is None:
            edges = array(vals[0]), array(vals[-1])
            vals = vals[1:-1]
        if shape is None:
            shape = avals[0].shape

        self.vals = vals
        self.shape = shape

        self.lower, self.upper = edges
        self.dtype = avals.dtype

    def __add__(self, other):
        return Domain(
            [v + other for v in self.vals],
            self.dtype,
            (self.lower + other, self.upper + other),
            self.shape,
        )

    def __mul__(self, other):
        try:
            return Domain(
                [v * other for v in self.vals],
                self.dtype,
                (self.lower * other, self.upper * other),
                self.shape,
            )
        except TypeError:
            return Domain(
                [v * other for v in self.vals],
                self.dtype,
                (self.lower, self.upper),
                self.shape,
            )

    def __neg__(self):
        return Domain([-v for v in self.vals], self.dtype, (-self.lower, -self.upper), self.shape)


def product(domains, n_samples=-1):
    """Get an iterator over a product of domains.

    Args:
        domains: a dictionary of (name, object) pairs, where the objects
                 must be "domain-like", as in, have a `.vals` property
        n_samples: int, maximum samples to return.  -1 to return whole product

    Returns:
        list of the cartesian product of the domains
    """
    try:
        names, domains = zip(*domains.items())
    except ValueError:  # domains.items() is empty
        return [{}]
    all_vals = [zip(names, val) for val in itertools.product(*[d.vals for d in domains])]
    if n_samples > 0 and len(all_vals) > n_samples:
        return (all_vals[j] for j in nr.choice(len(all_vals), n_samples, replace=False))
    return all_vals


R = Domain([-inf, -2.1, -1, -0.01, 0.0, 0.01, 1, 2.1, inf])
Rplus = Domain([0, 0.01, 0.1, 0.9, 0.99, 1, 1.5, 2, 100, inf])
Rplusbig = Domain([0, 0.5, 0.9, 0.99, 1, 1.5, 2, 20, inf])
Rminusbig = Domain([-inf, -2, -1.5, -1, -0.99, -0.9, -0.5, -0.01, 0])
Unit = Domain([0, 0.001, 0.1, 0.5, 0.75, 0.99, 1])

Circ = Domain([-np.pi, -2.1, -1, -0.01, 0.0, 0.01, 1, 2.1, np.pi])

Runif = Domain([-1, -0.4, 0, 0.4, 1])
Rdunif = Domain([-10, 0, 10.0])
Rplusunif = Domain([0, 0.5, inf])
Rplusdunif = Domain([2, 10, 100], "int64")

I = Domain([-1000, -3, -2, -1, 0, 1, 2, 3, 1000], "int64")

NatSmall = Domain([0, 3, 4, 5, 1000], "int64")
Nat = Domain([0, 1, 2, 3, 2000], "int64")
NatBig = Domain([0, 1, 2, 3, 5000, 50000], "int64")
PosNat = Domain([1, 2, 3, 2000], "int64")

Bool = Domain([0, 0, 1, 1], "int64")


def build_model(distfam, valuedomain, vardomains, extra_args=None):
    if extra_args is None:
        extra_args = {}
    with Model() as m:
        vals = {}
        for v, dom in vardomains.items():
            vals[v] = Flat(v, dtype=dom.dtype, shape=dom.shape, testval=dom.vals[0])
        vals.update(extra_args)
        distfam("value", shape=valuedomain.shape, transform=None, **vals)
    return m


def laplace_asymmetric_logpdf(value, kappa, b, mu):
    kapinv = 1 / kappa
    value = value - mu
    lPx = value * b * np.where(value >= 0, -kappa, kapinv)
    lPx += np.log(b / (kappa + kapinv))
    return lPx


def integrate_nd(f, domain, shape, dtype):
    if shape == () or shape == (1,):
        if dtype in continuous_types:
            return integrate.quad(f, domain.lower, domain.upper, epsabs=1e-8)[0]
        else:
            return sum(f(j) for j in range(domain.lower, domain.upper + 1))
    elif shape == (2,):

        def f2(a, b):
            return f([a, b])

        return integrate.dblquad(
            f2,
            domain.lower[0],
            domain.upper[0],
            lambda _: domain.lower[1],
            lambda _: domain.upper[1],
        )[0]
    elif shape == (3,):

        def f3(a, b, c):
            return f([a, b, c])

        return integrate.tplquad(
            f3,
            domain.lower[0],
            domain.upper[0],
            lambda _: domain.lower[1],
            lambda _: domain.upper[1],
            lambda _, __: domain.lower[2],
            lambda _, __: domain.upper[2],
        )[0]
    else:
        raise ValueError("Dont know how to integrate shape: " + str(shape))


def multinomial_logpdf(value, n, p):
    if value.sum() == n and (0 <= value).all() and (value <= n).all():
        logpdf = scipy.special.gammaln(n + 1)
        logpdf -= scipy.special.gammaln(value + 1).sum()
        logpdf += logpow(p, value).sum()
        return logpdf
    else:
        return -inf


def dirichlet_multinomial_logpmf(value, n, a):
    value, n, a = (np.asarray(x) for x in [value, n, a])
    assert value.ndim == 1
    assert n.ndim == 0
    assert a.shape == value.shape
    gammaln = scipy.special.gammaln
    if value.sum() == n and (0 <= value).all() and (value <= n).all():
        sum_a = a.sum(axis=-1)
        const = gammaln(n + 1) + gammaln(sum_a) - gammaln(n + sum_a)
        series = gammaln(value + a) - gammaln(value + 1) - gammaln(a)
        return const + series.sum(axis=-1)
    else:
        return -inf


def beta_mu_sigma(value, mu, sigma):
    kappa = mu * (1 - mu) / sigma ** 2 - 1
    if kappa > 0:
        return sp.beta.logpdf(value, mu * kappa, (1 - mu) * kappa)
    else:
        return -inf


class ProductDomain:
    def __init__(self, domains):
        self.vals = list(itertools.product(*[d.vals for d in domains]))
        self.shape = (len(domains),) + domains[0].shape
        self.lower = [d.lower for d in domains]
        self.upper = [d.upper for d in domains]
        self.dtype = domains[0].dtype


def Vector(D, n):
    return ProductDomain([D] * n)


def SortedVector(n):
    vals = []
    np.random.seed(42)
    for _ in range(10):
        vals.append(np.sort(np.random.randn(n)))
    return Domain(vals, edges=(None, None))


def UnitSortedVector(n):
    vals = []
    np.random.seed(42)
    for _ in range(10):
        vals.append(np.sort(np.random.rand(n)))
    return Domain(vals, edges=(None, None))


def RealMatrix(n, m):
    vals = []
    np.random.seed(42)
    for _ in range(10):
        vals.append(np.random.randn(n, m))
    return Domain(vals, edges=(None, None))


def simplex_values(n):
    if n == 1:
        yield array([1.0])
    else:
        for v in Unit.vals:
            for vals in simplex_values(n - 1):
                yield np.concatenate([[v], (1 - v) * vals])


def normal_logpdf_tau(value, mu, tau):
    return normal_logpdf_cov(value, mu, np.linalg.inv(tau)).sum()


def normal_logpdf_cov(value, mu, cov):
    return scipy.stats.multivariate_normal.logpdf(value, mu, cov).sum()


def normal_logpdf_chol(value, mu, chol):
    return normal_logpdf_cov(value, mu, np.dot(chol, chol.T)).sum()


def normal_logpdf_chol_upper(value, mu, chol):
    return normal_logpdf_cov(value, mu, np.dot(chol.T, chol)).sum()


def matrix_normal_logpdf_cov(value, mu, rowcov, colcov):
    return scipy.stats.matrix_normal.logpdf(value, mu, rowcov, colcov)


def matrix_normal_logpdf_chol(value, mu, rowchol, colchol):
    return matrix_normal_logpdf_cov(
        value, mu, np.dot(rowchol, rowchol.T), np.dot(colchol, colchol.T)
    )


def kron_normal_logpdf_cov(value, mu, covs, sigma):
    cov = kronecker(*covs).eval()
    if sigma is not None:
        cov += sigma ** 2 * np.eye(*cov.shape)
    return scipy.stats.multivariate_normal.logpdf(value, mu, cov).sum()


def kron_normal_logpdf_chol(value, mu, chols, sigma):
    covs = [np.dot(chol, chol.T) for chol in chols]
    return kron_normal_logpdf_cov(value, mu, covs, sigma=sigma)


def kron_normal_logpdf_evd(value, mu, evds, sigma):
    covs = []
    for eigs, Q in evds:
        try:
            eigs = eigs.eval()
        except AttributeError:
            pass
        try:
            Q = Q.eval()
        except AttributeError:
            pass
        covs.append(np.dot(Q, np.dot(np.diag(eigs), Q.T)))
    return kron_normal_logpdf_cov(value, mu, covs, sigma)


def betafn(a):
    return floatX(scipy.special.gammaln(a).sum(-1) - scipy.special.gammaln(a.sum(-1)))


def logpow(v, p):
    return np.choose(v == 0, [p * np.log(v), 0])


def discrete_weibull_logpmf(value, q, beta):
    return floatX(
        np.log(
            np.power(floatX(q), np.power(floatX(value), floatX(beta)))
            - np.power(floatX(q), np.power(floatX(value + 1), floatX(beta)))
        )
    )


def dirichlet_logpdf(value, a):
    return floatX((-betafn(a) + logpow(value, a - 1).sum(-1)).sum())


def categorical_logpdf(value, p):
    if value >= 0 and value <= len(p):
        return floatX(np.log(np.moveaxis(p, -1, 0)[value]))
    else:
        return -inf


def mvt_logpdf(value, nu, Sigma, mu=0):
    d = len(Sigma)
    dist = np.atleast_2d(value) - mu
    chol = np.linalg.cholesky(Sigma)
    trafo = np.linalg.solve(chol, dist.T).T
    logdet = np.log(np.diag(chol)).sum()

    lgamma = scipy.special.gammaln
    norm = lgamma((nu + d) / 2.0) - 0.5 * d * np.log(nu * np.pi) - lgamma(nu / 2.0)
    logp = norm - logdet - (nu + d) / 2.0 * np.log1p((trafo * trafo).sum(-1) / nu)
    return logp.sum()


def AR1_logpdf(value, k, tau_e):
    tau = tau_e * (1 - k ** 2)
    return (
        sp.norm(loc=0, scale=1 / np.sqrt(tau)).logpdf(value[0])
        + sp.norm(loc=k * value[:-1], scale=1 / np.sqrt(tau_e)).logpdf(value[1:]).sum()
    )


def invlogit(x, eps=sys.float_info.epsilon):
    return (1.0 - 2.0 * eps) / (1.0 + np.exp(-x)) + eps


def orderedlogistic_logpdf(value, eta, cutpoints):
    c = np.concatenate(([-np.inf], cutpoints, [np.inf]))
    ps = np.array([invlogit(eta - cc) - invlogit(eta - cc1) for cc, cc1 in zip(c[:-1], c[1:])])
    p = ps[value]
    return np.where(np.all(ps >= 0), np.log(p), -np.inf)


def invprobit(x):
    return (erf(x / np.sqrt(2)) + 1) / 2


def orderedprobit_logpdf(value, eta, cutpoints):
    c = np.concatenate(([-np.inf], cutpoints, [np.inf]))
    ps = np.array([invprobit(eta - cc) - invprobit(eta - cc1) for cc, cc1 in zip(c[:-1], c[1:])])
    p = ps[value]
    return np.where(np.all(ps >= 0), np.log(p), -np.inf)


class Simplex:
    def __init__(self, n):
        self.vals = list(simplex_values(n))
        self.shape = (n,)
        self.dtype = Unit.dtype


class MultiSimplex:
    def __init__(self, n_dependent, n_independent):
        self.vals = []
        for simplex_value in itertools.product(simplex_values(n_dependent), repeat=n_independent):
            self.vals.append(np.vstack(simplex_value))
        self.shape = (n_independent, n_dependent)
        self.dtype = Unit.dtype


def PdMatrix(n):
    if n == 1:
        return PdMatrix1
    elif n == 2:
        return PdMatrix2
    elif n == 3:
        return PdMatrix3
    else:
        raise ValueError("n out of bounds")


PdMatrix1 = Domain([np.eye(1), [[0.5]]], edges=(None, None))

PdMatrix2 = Domain([np.eye(2), [[0.5, 0.05], [0.05, 4.5]]], edges=(None, None))

PdMatrix3 = Domain([np.eye(3), [[0.5, 0.1, 0], [0.1, 1, 0], [0, 0, 2.5]]], edges=(None, None))


PdMatrixChol1 = Domain([np.eye(1), [[0.001]]], edges=(None, None))
PdMatrixChol2 = Domain([np.eye(2), [[0.1, 0], [10, 1]]], edges=(None, None))
PdMatrixChol3 = Domain([np.eye(3), [[0.1, 0, 0], [10, 100, 0], [0, 1, 10]]], edges=(None, None))


def PdMatrixChol(n):
    if n == 1:
        return PdMatrixChol1
    elif n == 2:
        return PdMatrixChol2
    elif n == 3:
        return PdMatrixChol3
    else:
        raise ValueError("n out of bounds")


PdMatrixCholUpper1 = Domain([np.eye(1), [[0.001]]], edges=(None, None))
PdMatrixCholUpper2 = Domain([np.eye(2), [[0.1, 10], [0, 1]]], edges=(None, None))
PdMatrixCholUpper3 = Domain(
    [np.eye(3), [[0.1, 10, 0], [0, 100, 1], [0, 0, 10]]], edges=(None, None)
)


def PdMatrixCholUpper(n):
    if n == 1:
        return PdMatrixCholUpper1
    elif n == 2:
        return PdMatrixCholUpper2
    elif n == 3:
        return PdMatrixCholUpper3
    else:
        raise ValueError("n out of bounds")


def RandomPdMatrix(n):
    A = np.random.rand(n, n)
    return np.dot(A, A.T) + n * np.identity(n)


class TestMatchesScipy:
    def check_logp(
        self,
        pymc3_dist,
        domain,
        paramdomains,
        scipy_logp,
        decimal=None,
        n_samples=100,
        extra_args=None,
        scipy_args=None,
    ):
        """
        Generic test for PyMC3 logp methods

        Test PyMC3 logp and equivalent scipy logpmf/logpdf methods give similar
        results for valid values and parameters inside the supported edges.
        Edges are excluded by default, but can be artificially included by
        creating a domain with repeated values (e.g., `Domain([0, 0, .5, 1, 1]`)

        Parameters
        ----------
        pymc3_dist: PyMC3 distribution
        domain : Domain
            Supported domain of distribution values
        paramdomains : Dictionary of Parameter : Domain pairs
            Supported domains of distribution parameters
        scipy_logp : Scipy logpmf/logpdf method
            Scipy logp method of equivalent pymc3_dist distribution
        decimal : Int
            Level of precision with which pymc3_dist and scipy logp are compared.
            Defaults to 6 for float64 and 3 for float32
        n_samples : Int
            Upper limit on the number of valid domain and value combinations that
            are compared between pymc3 and scipy methods. If n_samples is below the
            total number of combinations, a random subset is evaluated. Setting
            n_samples = -1, will return all possible combinations. Defaults to 100
        extra_args : Dictionary with extra arguments needed to build pymc3 model
            Dictionary is passed to helper function `build_model` from which
            the pymc3 distribution logp is calculated
        scipy_args : Dictionary with extra arguments needed to call scipy logp method
            Usually the same as extra_args
        """
        if decimal is None:
            decimal = select_by_precision(float64=6, float32=3)

        if extra_args is None:
            extra_args = {}

        if scipy_args is None:
            scipy_args = {}

        def logp_reference(args):
            args.update(scipy_args)
            return scipy_logp(**args)

        model = build_model(pymc3_dist, domain, paramdomains, extra_args)
        logp = model.fastlogp

        domains = paramdomains.copy()
        domains["value"] = domain
        for pt in product(domains, n_samples=n_samples):
            pt = Point(pt, model=model)
            assert_almost_equal(
                logp(pt),
                logp_reference(pt),
                decimal=decimal,
                err_msg=str(pt),
            )

    def check_logcdf(
        self,
        pymc3_dist,
        domain,
        paramdomains,
        scipy_logcdf,
        decimal=None,
        n_samples=100,
        skip_paramdomain_inside_edge_test=False,
        skip_paramdomain_outside_edge_test=False,
    ):
        """
        Generic test for PyMC3 logcdf methods

        The following tests are performed by default:
            1. Test PyMC3 logcdf and equivalent scipy logcdf methods give similar
            results for valid values and parameters inside the supported edges.
            Edges are excluded by default, but can be artificially included by
            creating a domain with repeated values (e.g., `Domain([0, 0, .5, 1, 1]`)
            Can be skipped via skip_paramdomain_inside_edge_test
            2. Test PyMC3 logcdf method returns -inf for invalid parameter values
            outside the supported edges. Can be skipped via skip_paramdomain_outside_edge_test
            3. Test PyMC3 logcdf method returns -inf and 0 for values below and
            above the supported edge, respectively, when using valid parameters.
            4. Test PyMC3 logcdf methods works with multiple value or returns
            default informative TypeError

        Parameters
        ----------
        pymc3_dist: PyMC3 distribution
        domain : Domain
            Supported domain of distribution values
        paramdomains : Dictionary of Parameter : Domain pairs
            Supported domains of distribution parameters
        scipy_logcdf : Scipy logcdf method
            Scipy logcdf method of equivalent pymc3_dist distribution
        decimal : Int
            Level of precision with which pymc3_dist and scipy_logcdf are compared.
            Defaults to 6 for float64 and 3 for float32
        n_samples : Int
            Upper limit on the number of valid domain and value combinations that
            are compared between pymc3 and scipy methods. If n_samples is below the
            total number of combinations, a random subset is evaluated. Setting
            n_samples = -1, will return all possible combinations. Defaults to 100
        skip_paramdomain_inside_edge_test : Bool
            Whether to run test 1., which checks that pymc3 and scipy distributions
            match for valid values and parameters inside the respective domain edges
        skip_paramdomain_outside_edge_test : Bool
            Whether to run test 2., which checks that pymc3 distribution logcdf
            returns -inf for invalid parameter values outside the supported domain edge

        Returns
        -------

        """
        # Test pymc3 and scipy distributions match for values and parameters
        # within the supported domain edges (excluding edges)
        if not skip_paramdomain_inside_edge_test:
            domains = paramdomains.copy()
            domains["value"] = domain
            if decimal is None:
                decimal = select_by_precision(float64=6, float32=3)
            for pt in product(domains, n_samples=n_samples):
                params = dict(pt)
                scipy_cdf = scipy_logcdf(**params)
                value = params.pop("value")
                dist = pymc3_dist.dist(**params)
                params["value"] = value  # for displaying in err_msg
                assert_almost_equal(
                    dist.logcdf(value).tag.test_value,
                    scipy_cdf,
                    decimal=decimal,
                    err_msg=str(params),
                )

        valid_value = domain.vals[0]
        valid_params = {param: paramdomain.vals[0] for param, paramdomain in paramdomains.items()}
        valid_dist = pymc3_dist.dist(**valid_params)

        # Natural domains do not have inf as the upper edge, but should also be ignored
        nat_domains = (NatSmall, Nat, NatBig, PosNat)

        # Test pymc3 distribution gives -inf for parameters outside the
        # supported domain edges (excluding edgse)
        if not skip_paramdomain_outside_edge_test:
            # Step1: collect potential invalid parameters
            invalid_params = {param: [None, None] for param in paramdomains}
            for param, paramdomain in paramdomains.items():
                if np.isfinite(paramdomain.lower):
                    invalid_params[param][0] = paramdomain.lower - 1
                if np.isfinite(paramdomain.upper) and paramdomain not in nat_domains:
                    invalid_params[param][1] = paramdomain.upper + 1
            # Step2: test invalid parameters, one a time
            for invalid_param, invalid_edges in invalid_params.items():
                for invalid_edge in invalid_edges:
                    if invalid_edge is not None:
                        test_params = valid_params.copy()  # Shallow copy should be okay
                        test_params[invalid_param] = invalid_edge
                        invalid_dist = pymc3_dist.dist(**test_params)
                        assert_equal(
                            invalid_dist.logcdf(valid_value).tag.test_value,
                            -np.inf,
                            err_msg=str(test_params),
                        )

        # Test that values below domain edge evaluate to -np.inf
        if np.isfinite(domain.lower):
            below_domain = domain.lower - 1
            assert_equal(
                valid_dist.logcdf(below_domain).tag.test_value,
                -np.inf,
                err_msg=str(below_domain),
            )

        # Test that values above domain edge evaluate to 0
        if domain not in nat_domains and np.isfinite(domain.upper):
            above_domain = domain.upper + 1
            assert_equal(
                valid_dist.logcdf(above_domain).tag.test_value,
                0,
                err_msg=str(above_domain),
            )

        # Test that method works with multiple values or raises informative TypeError
        try:
            valid_dist.logcdf(np.array([valid_value, valid_value])).tag.test_value
        except TypeError as err:
            if not str(err).endswith(
                ".logcdf expects a scalar value but received a 1-dimensional object."
            ):
                raise

    def check_selfconsistency_discrete_logcdf(
        self, distribution, domain, paramdomains, decimal=None, n_samples=100
    ):
        """
        Check that logcdf of discrete distributions matches sum of logps up to value
        """
        domains = paramdomains.copy()
        domains["value"] = domain
        if decimal is None:
            decimal = select_by_precision(float64=6, float32=3)
        for pt in product(domains, n_samples=n_samples):
            params = dict(pt)
            value = params.pop("value")
            values = np.arange(domain.lower, value + 1)
            dist = distribution.dist(**params)
            assert_almost_equal(
                dist.logcdf(value).tag.test_value,
                logsumexp(dist.logp(values), keepdims=False).tag.test_value,
                decimal=decimal,
                err_msg=str(pt),
            )

    def check_int_to_1(self, model, value, domain, paramdomains):
        pdf = model.fastfn(exp(model.logpt))
        for pt in product(paramdomains, n_samples=10):
            pt = Point(pt, value=value.tag.test_value, model=model)
            bij = DictToVarBijection(value, (), pt)
            pdfx = bij.mapf(pdf)
            area = integrate_nd(pdfx, domain, value.dshape, value.dtype)
            assert_almost_equal(area, 1, err_msg=str(pt))

    def checkd(self, distfam, valuedomain, vardomains, checks=None, extra_args=None):
        if checks is None:
            checks = (self.check_int_to_1,)

        if extra_args is None:
            extra_args = {}
        m = build_model(distfam, valuedomain, vardomains, extra_args=extra_args)
        for check in checks:
            check(m, m.named_vars["value"], valuedomain, vardomains)

    def test_uniform(self):
        self.check_logp(
            Uniform,
            Runif,
            {"lower": -Rplusunif, "upper": Rplusunif},
            lambda value, lower, upper: sp.uniform.logpdf(value, lower, upper - lower),
        )
        self.check_logcdf(
            Uniform,
            Runif,
            {"lower": -Rplusunif, "upper": Rplusunif},
            lambda value, lower, upper: sp.uniform.logcdf(value, lower, upper - lower),
            skip_paramdomain_outside_edge_test=True,
        )
        # Custom logp / logcdf check for invalid parameters
        invalid_dist = Uniform.dist(lower=1, upper=0)
        assert invalid_dist.logp(0.5).tag.test_value == -np.inf
        assert invalid_dist.logcdf(2).tag.test_value == -np.inf

    def test_triangular(self):
        self.check_logp(
            Triangular,
            Runif,
            {"lower": -Rplusunif, "c": Runif, "upper": Rplusunif},
            lambda value, c, lower, upper: sp.triang.logpdf(value, c - lower, lower, upper - lower),
        )
        self.check_logcdf(
            Triangular,
            Runif,
            {"lower": -Rplusunif, "c": Runif, "upper": Rplusunif},
            lambda value, c, lower, upper: sp.triang.logcdf(value, c - lower, lower, upper - lower),
            skip_paramdomain_outside_edge_test=True,
        )
        # Custom logp check for invalid value
        valid_dist = Triangular.dist(lower=0, upper=1, c=2.0)
        assert np.all(valid_dist.logp(np.array([1.9, 2.0, 2.1])).tag.test_value == -np.inf)

        # Custom logp / logcdf check for invalid parameters
        invalid_dist = Triangular.dist(lower=1, upper=0, c=2.0)
        assert invalid_dist.logp(0.5).tag.test_value == -np.inf
        assert invalid_dist.logcdf(2).tag.test_value == -np.inf

    def test_bound_normal(self):
        PositiveNormal = Bound(Normal, lower=0.0)
        self.check_logp(
            PositiveNormal,
            Rplus,
            {"mu": Rplus, "sigma": Rplus},
            lambda value, mu, sigma: sp.norm.logpdf(value, mu, sigma),
            decimal=select_by_precision(float64=6, float32=-1),
        )
        with Model():
            x = PositiveNormal("x", mu=0, sigma=1, transform=None)
        assert np.isinf(x.logp({"x": -1}))

    def test_discrete_unif(self):
        self.check_logp(
            DiscreteUniform,
            Rdunif,
            {"lower": -Rplusdunif, "upper": Rplusdunif},
            lambda value, lower, upper: sp.randint.logpmf(value, lower, upper + 1),
        )
        self.check_logcdf(
            DiscreteUniform,
            Rdunif,
            {"lower": -Rplusdunif, "upper": Rplusdunif},
            lambda value, lower, upper: sp.randint.logcdf(value, lower, upper + 1),
            skip_paramdomain_outside_edge_test=True,
        )
        self.check_selfconsistency_discrete_logcdf(
            DiscreteUniform,
            Rdunif,
            {"lower": -Rplusdunif, "upper": Rplusdunif},
        )
        # Custom logp / logcdf check for invalid parameters
        invalid_dist = DiscreteUniform.dist(lower=1, upper=0)
        assert invalid_dist.logp(0.5).tag.test_value == -np.inf
        assert invalid_dist.logcdf(2).tag.test_value == -np.inf

    def test_flat(self):
        self.check_logp(Flat, Runif, {}, lambda value: 0)
        with Model():
            x = Flat("a")
            assert_allclose(x.tag.test_value, 0)
        self.check_logcdf(Flat, R, {}, lambda value: np.log(0.5))
        # Check infinite cases individually.
        assert 0.0 == Flat.dist().logcdf(np.inf).tag.test_value
        assert -np.inf == Flat.dist().logcdf(-np.inf).tag.test_value

    def test_half_flat(self):
        self.check_logp(HalfFlat, Rplus, {}, lambda value: 0)
        with Model():
            x = HalfFlat("a", shape=2)
            assert_allclose(x.tag.test_value, 1)
            assert x.tag.test_value.shape == (2,)
        self.check_logcdf(HalfFlat, Rplus, {}, lambda value: -np.inf)
        # Check infinite cases individually.
        assert 0.0 == HalfFlat.dist().logcdf(np.inf).tag.test_value
        assert -np.inf == HalfFlat.dist().logcdf(-np.inf).tag.test_value

    def test_normal(self):
        self.check_logp(
            Normal,
            R,
            {"mu": R, "sigma": Rplus},
            lambda value, mu, sigma: sp.norm.logpdf(value, mu, sigma),
            decimal=select_by_precision(float64=6, float32=1),
        )
        self.check_logcdf(
            Normal,
            R,
            {"mu": R, "sigma": Rplus},
            lambda value, mu, sigma: sp.norm.logcdf(value, mu, sigma),
        )

    def test_truncated_normal(self):
        def scipy_logp(value, mu, sigma, lower, upper):
            return sp.truncnorm.logpdf(
                value, (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
            )

        self.check_logp(
            TruncatedNormal,
            R,
            {"mu": R, "sigma": Rplusbig, "lower": -Rplusbig, "upper": Rplusbig},
            scipy_logp,
            decimal=select_by_precision(float64=6, float32=1),
        )

    def test_half_normal(self):
        self.check_logp(
            HalfNormal,
            Rplus,
            {"sigma": Rplus},
            lambda value, sigma: sp.halfnorm.logpdf(value, scale=sigma),
            decimal=select_by_precision(float64=6, float32=-1),
        )
        self.check_logcdf(
            HalfNormal,
            Rplus,
            {"sigma": Rplus},
            lambda value, sigma: sp.halfnorm.logcdf(value, scale=sigma),
        )

    def test_chi_squared(self):
        self.check_logp(
            ChiSquared,
            Rplus,
            {"nu": Rplusdunif},
            lambda value, nu: sp.chi2.logpdf(value, df=nu),
        )

    @pytest.mark.xfail(
        condition=(theano.config.floatX == "float32"),
        reason="Poor CDF in SciPy. See scipy/scipy#869 for details.",
    )
    def test_wald_scipy(self):
        self.check_logp(
            Wald,
            Rplus,
            {"mu": Rplus, "alpha": Rplus},
            lambda value, mu, alpha: sp.invgauss.logpdf(value, mu=mu, loc=alpha),
            decimal=select_by_precision(float64=6, float32=1),
        )
        self.check_logcdf(
            Wald,
            Rplus,
            {"mu": Rplus, "alpha": Rplus},
            lambda value, mu, alpha: sp.invgauss.logcdf(value, mu=mu, loc=alpha),
        )

    @pytest.mark.parametrize(
        "value,mu,lam,phi,alpha,logp",
        [
            (0.5, 0.001, 0.5, None, 0.0, -124500.7257914),
            (1.0, 0.5, 0.001, None, 0.0, -4.3733162),
            (2.0, 1.0, None, None, 0.0, -2.2086593),
            (5.0, 2.0, 2.5, None, 0.0, -3.4374500),
            (7.5, 5.0, None, 1.0, 0.0, -3.2199074),
            (15.0, 10.0, None, 0.75, 0.0, -4.0360623),
            (50.0, 15.0, None, 0.66666, 0.0, -6.1801249),
            (0.5, 0.001, 0.5, None, 0.0, -124500.7257914),
            (1.0, 0.5, 0.001, None, 0.5, -3.3330954),
            (2.0, 1.0, None, None, 1.0, -0.9189385),
            (5.0, 2.0, 2.5, None, 2.0, -2.2128783),
            (7.5, 5.0, None, 1.0, 2.5, -2.5283764),
            (15.0, 10.0, None, 0.75, 5.0, -3.3653647),
            (50.0, 15.0, None, 0.666666, 10.0, -5.6481874),
        ],
    )
    def test_wald(self, value, mu, lam, phi, alpha, logp):
        # Log probabilities calculated using the dIG function from the R package gamlss.
        # See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or
        # http://www.gamlss.org/.
        with Model() as model:
            Wald("wald", mu=mu, lam=lam, phi=phi, alpha=alpha, transform=None)
        pt = {"wald": value}
        decimals = select_by_precision(float64=6, float32=1)
        assert_almost_equal(model.fastlogp(pt), logp, decimal=decimals, err_msg=str(pt))

    def test_beta(self):
        self.check_logp(
            Beta,
            Unit,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: sp.beta.logpdf(value, alpha, beta),
        )
        self.check_logp(Beta, Unit, {"mu": Unit, "sigma": Rplus}, beta_mu_sigma)
        self.check_logcdf(
            Beta,
            Unit,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: sp.beta.logcdf(value, alpha, beta),
            n_samples=10,
        )

    def test_kumaraswamy(self):
        # Scipy does not have a built-in Kumaraswamy pdf
        def scipy_log_pdf(value, a, b):
            return (
                np.log(a) + np.log(b) + (a - 1) * np.log(value) + (b - 1) * np.log(1 - value ** a)
            )

        self.check_logp(Kumaraswamy, Unit, {"a": Rplus, "b": Rplus}, scipy_log_pdf)

    def test_exponential(self):
        self.check_logp(
            Exponential,
            Rplus,
            {"lam": Rplus},
            lambda value, lam: sp.expon.logpdf(value, 0, 1 / lam),
        )
        self.check_logcdf(
            Exponential,
            Rplus,
            {"lam": Rplus},
            lambda value, lam: sp.expon.logcdf(value, 0, 1 / lam),
        )

    def test_geometric(self):
        self.check_logp(
            Geometric,
            Nat,
            {"p": Unit},
            lambda value, p: np.log(sp.geom.pmf(value, p)),
        )
        self.check_logcdf(
            Geometric,
            Nat,
            {"p": Unit},
            lambda value, p: sp.geom.logcdf(value, p),
        )
        self.check_selfconsistency_discrete_logcdf(
            Geometric,
            Nat,
            {"p": Unit},
        )

    def test_hypergeometric(self):
        def modified_scipy_hypergeom_logpmf(value, N, k, n):
            # Convert nan to -np.inf
            original_res = sp.hypergeom.logpmf(value, N, k, n)
            return original_res if not np.isnan(original_res) else -np.inf

        def modified_scipy_hypergeom_logcdf(value, N, k, n):
            # Convert nan to -np.inf
            original_res = sp.hypergeom.logcdf(value, N, k, n)

            # Correct for scipy bug in logcdf method (see https://github.com/scipy/scipy/issues/13280)
            if not np.isnan(original_res):
                pmfs = sp.hypergeom.logpmf(np.arange(value + 1), N, k, n)
                if np.all(np.isnan(pmfs)):
                    original_res = np.nan

            return original_res if not np.isnan(original_res) else -np.inf

        self.check_logp(
            HyperGeometric,
            Nat,
            {"N": NatSmall, "k": NatSmall, "n": NatSmall},
            modified_scipy_hypergeom_logpmf,
        )
        self.check_logcdf(
            HyperGeometric,
            Nat,
            {"N": NatSmall, "k": NatSmall, "n": NatSmall},
            modified_scipy_hypergeom_logcdf,
        )
        self.check_selfconsistency_discrete_logcdf(
            HyperGeometric,
            Nat,
            {"N": NatSmall, "k": NatSmall, "n": NatSmall},
        )

    @pytest.mark.xfail(
        condition=(theano.config.floatX == "float32"),
        reason="Fails on float32 due to a float casting problem in the logcdf",
    )
    def test_negative_binomial(self):
        def scipy_mu_alpha_logpmf(value, mu, alpha):
            return sp.nbinom.logpmf(value, alpha, 1 - mu / (mu + alpha))

        def scipy_mu_alpha_logcdf(value, mu, alpha):
            return sp.nbinom.logcdf(value, alpha, 1 - mu / (mu + alpha))

        self.check_logp(
            NegativeBinomial,
            Nat,
            {"mu": Rplus, "alpha": Rplus},
            scipy_mu_alpha_logpmf,
        )
        self.check_logp(
            NegativeBinomial,
            Nat,
            {"p": Unit, "n": Rplus},
            lambda value, p, n: sp.nbinom.logpmf(value, n, p),
        )
        self.check_logcdf(
            NegativeBinomial,
            Nat,
            {"mu": Rplus, "alpha": Rplus},
            scipy_mu_alpha_logcdf,
            n_samples=5,
        )
        self.check_logcdf(
            NegativeBinomial,
            Nat,
            {"p": Unit, "n": Rplus},
            lambda value, p, n: sp.nbinom.logcdf(value, n, p),
            n_samples=5,
        )
        self.check_selfconsistency_discrete_logcdf(
            NegativeBinomial,
            Nat,
            {"mu": Rplus, "alpha": Rplus},
            n_samples=10,
        )

    @pytest.mark.parametrize(
        "mu, p, alpha, n, expected",
        [
            (5, None, None, None, "Must specify either alpha or n."),
            (None, 0.5, None, None, "Must specify either alpha or n."),
            (None, None, None, None, "Must specify either alpha or n."),
            (5, None, 2, 2, "Can't specify both alpha and n."),
            (None, 0.5, 2, 2, "Can't specify both alpha and n."),
            (None, None, 2, 2, "Can't specify both alpha and n."),
            (None, None, 2, None, "Must specify either mu or p."),
            (None, None, None, 2, "Must specify either mu or p."),
            (5, 0.5, 2, None, "Can't specify both mu and p."),
            (5, 0.5, None, 2, "Can't specify both mu and p."),
        ],
    )
    def test_negative_binomial_init_fail(self, mu, p, alpha, n, expected):
        with Model():
            with pytest.raises(ValueError, match=f"Incompatible parametrization. {expected}"):
                NegativeBinomial("x", mu=mu, p=p, alpha=alpha, n=n)

    def test_laplace(self):
        self.check_logp(
            Laplace,
            R,
            {"mu": R, "b": Rplus},
            lambda value, mu, b: sp.laplace.logpdf(value, mu, b),
        )
        self.check_logcdf(
            Laplace,
            R,
            {"mu": R, "b": Rplus},
            lambda value, mu, b: sp.laplace.logcdf(value, mu, b),
        )

    def test_laplace_asymmetric(self):
        self.check_logp(
            AsymmetricLaplace,
            R,
            {"b": Rplus, "kappa": Rplus, "mu": R},
            laplace_asymmetric_logpdf,
        )

    def test_lognormal(self):
        self.check_logp(
            Lognormal,
            Rplus,
            {"mu": R, "tau": Rplusbig},
            lambda value, mu, tau: floatX(sp.lognorm.logpdf(value, tau ** -0.5, 0, np.exp(mu))),
        )
        self.check_logcdf(
            Lognormal,
            Rplus,
            {"mu": R, "tau": Rplusbig},
            lambda value, mu, tau: sp.lognorm.logcdf(value, tau ** -0.5, 0, np.exp(mu)),
        )

    def test_t(self):
        self.check_logp(
            StudentT,
            R,
            {"nu": Rplus, "mu": R, "lam": Rplus},
            lambda value, nu, mu, lam: sp.t.logpdf(value, nu, mu, lam ** -0.5),
        )
        self.check_logcdf(
            StudentT,
            R,
            {"nu": Rplus, "mu": R, "lam": Rplus},
            lambda value, nu, mu, lam: sp.t.logcdf(value, nu, mu, lam ** -0.5),
            n_samples=10,
        )

    def test_cauchy(self):
        self.check_logp(
            Cauchy,
            R,
            {"alpha": R, "beta": Rplusbig},
            lambda value, alpha, beta: sp.cauchy.logpdf(value, alpha, beta),
        )
        self.check_logcdf(
            Cauchy,
            R,
            {"alpha": R, "beta": Rplusbig},
            lambda value, alpha, beta: sp.cauchy.logcdf(value, alpha, beta),
        )

    def test_half_cauchy(self):
        self.check_logp(
            HalfCauchy,
            Rplus,
            {"beta": Rplusbig},
            lambda value, beta: sp.halfcauchy.logpdf(value, scale=beta),
        )
        self.check_logcdf(
            HalfCauchy,
            Rplus,
            {"beta": Rplusbig},
            lambda value, beta: sp.halfcauchy.logcdf(value, scale=beta),
        )

    def test_gamma_logp(self):
        self.check_logp(
            Gamma,
            Rplus,
            {"alpha": Rplusbig, "beta": Rplusbig},
            lambda value, alpha, beta: sp.gamma.logpdf(value, alpha, scale=1.0 / beta),
        )

        def test_fun(value, mu, sigma):
            return sp.gamma.logpdf(value, mu ** 2 / sigma ** 2, scale=1.0 / (mu / sigma ** 2))

        self.check_logp(
            Gamma,
            Rplus,
            {"mu": Rplusbig, "sigma": Rplusbig},
            test_fun,
        )

    @pytest.mark.xfail(
        condition=(theano.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_gamma_logcdf(self):
        # pymc-devs/Theano-PyMC#224: skip_paramdomain_outside_edge_test has to be set
        # True to avoid triggering a C-level assertion in the Theano GammaQ function
        # in gamma.c file. Can be set back to False (default) once that issue is solved
        self.check_logcdf(
            Gamma,
            Rplus,
            {"alpha": Rplusbig, "beta": Rplusbig},
            lambda value, alpha, beta: sp.gamma.logcdf(value, alpha, scale=1.0 / beta),
            skip_paramdomain_outside_edge_test=True,
        )

    @pytest.mark.xfail(
        condition=(theano.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_inverse_gamma(self):
        self.check_logp(
            InverseGamma,
            Rplus,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: sp.invgamma.logpdf(value, alpha, scale=beta),
        )
        # pymc-devs/Theano-PyMC#224: skip_paramdomain_outside_edge_test has to be set
        # True to avoid triggering a C-level assertion in the Theano GammaQ function
        # in gamma.c file. Can be set back to False (default) once that issue is solved
        self.check_logcdf(
            InverseGamma,
            Rplus,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: sp.invgamma.logcdf(value, alpha, scale=beta),
            skip_paramdomain_outside_edge_test=True,
        )

    @pytest.mark.xfail(
        condition=(theano.config.floatX == "float32"),
        reason="Fails on float32 due to scaling issues",
    )
    def test_inverse_gamma_alt_params(self):
        def test_fun(value, mu, sigma):
            alpha, beta = InverseGamma._get_alpha_beta(None, None, mu, sigma)
            return sp.invgamma.logpdf(value, alpha, scale=beta)

        self.check_logp(
            InverseGamma,
            Rplus,
            {"mu": Rplus, "sigma": Rplus},
            test_fun,
            decimal=select_by_precision(float64=5, float32=3),
        )

    def test_pareto(self):
        self.check_logp(
            Pareto,
            Rplus,
            {"alpha": Rplusbig, "m": Rplusbig},
            lambda value, alpha, m: sp.pareto.logpdf(value, alpha, scale=m),
        )
        self.check_logcdf(
            Pareto,
            Rplus,
            {"alpha": Rplusbig, "m": Rplusbig},
            lambda value, alpha, m: sp.pareto.logcdf(value, alpha, scale=m),
        )

    @pytest.mark.xfail(
        condition=(theano.config.floatX == "float32"),
        reason="Fails on float32 due to inf issues",
    )
    def test_weibull(self):
        self.check_logp(
            Weibull,
            Rplus,
            {"alpha": Rplusbig, "beta": Rplusbig},
            lambda value, alpha, beta: sp.exponweib.logpdf(value, 1, alpha, scale=beta),
        )
        self.check_logcdf(
            Weibull,
            Rplus,
            {"alpha": Rplusbig, "beta": Rplusbig},
            lambda value, alpha, beta: sp.exponweib.logcdf(value, 1, alpha, scale=beta),
        )

    def test_half_studentt(self):
        # this is only testing for nu=1 (halfcauchy)
        self.check_logp(
            HalfStudentT,
            Rplus,
            {"sigma": Rplus},
            lambda value, sigma: sp.halfcauchy.logpdf(value, 0, sigma),
        )

    def test_skew_normal(self):
        self.check_logp(
            SkewNormal,
            R,
            {"mu": R, "sigma": Rplusbig, "alpha": R},
            lambda value, alpha, mu, sigma: sp.skewnorm.logpdf(value, alpha, mu, sigma),
            decimal=select_by_precision(float64=5, float32=3),
        )

    def test_binomial(self):
        self.check_logp(
            Binomial,
            Nat,
            {"n": NatSmall, "p": Unit},
            lambda value, n, p: sp.binom.logpmf(value, n, p),
        )
        self.check_logcdf(
            Binomial,
            Nat,
            {"n": NatSmall, "p": Unit},
            lambda value, n, p: sp.binom.logcdf(value, n, p),
            n_samples=10,
        )
        self.check_selfconsistency_discrete_logcdf(
            Binomial,
            Nat,
            {"n": NatSmall, "p": Unit},
            n_samples=10,
        )

    # Too lazy to propagate decimal parameter through the whole chain of deps
    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    @pytest.mark.xfail(
        condition=(SCIPY_VERSION < parse("1.4.0")), reason="betabinom is new in Scipy 1.4.0"
    )
    def test_beta_binomial(self):
        self.checkd(
            BetaBinomial,
            Nat,
            {"alpha": Rplus, "beta": Rplus, "n": NatSmall},
        )
        self.check_logp(
            BetaBinomial,
            Nat,
            {"alpha": Rplus, "beta": Rplus, "n": NatSmall},
            lambda value, alpha, beta, n: sp.betabinom.logpmf(value, a=alpha, b=beta, n=n),
        )
        self.check_logcdf(
            BetaBinomial,
            Nat,
            {"alpha": Rplus, "beta": Rplus, "n": NatSmall},
            lambda value, alpha, beta, n: sp.betabinom.logcdf(value, a=alpha, b=beta, n=n),
        )
        self.check_selfconsistency_discrete_logcdf(
            BetaBinomial,
            Nat,
            {"alpha": Rplus, "beta": Rplus, "n": NatSmall},
        )

    def test_bernoulli(self):
        self.check_logp(
            Bernoulli,
            Bool,
            {"logit_p": R},
            lambda value, logit_p: sp.bernoulli.logpmf(value, scipy.special.expit(logit_p)),
        )
        self.check_logp(
            Bernoulli,
            Bool,
            {"p": Unit},
            lambda value, p: sp.bernoulli.logpmf(value, p),
        )
        self.check_logcdf(
            Bernoulli,
            Bool,
            {"p": Unit},
            lambda value, p: sp.bernoulli.logcdf(value, p),
        )
        self.check_logcdf(
            Bernoulli,
            Bool,
            {"logit_p": R},
            lambda value, logit_p: sp.bernoulli.logcdf(value, scipy.special.expit(logit_p)),
        )
        self.check_selfconsistency_discrete_logcdf(
            Bernoulli,
            Bool,
            {"p": Unit},
        )

    def test_discrete_weibull(self):
        self.check_logp(
            DiscreteWeibull,
            Nat,
            {"q": Unit, "beta": Rplusdunif},
            discrete_weibull_logpmf,
        )
        self.check_selfconsistency_discrete_logcdf(
            DiscreteWeibull,
            Nat,
            {"q": Unit, "beta": Rplusdunif},
        )

    def test_poisson(self):
        self.check_logp(
            Poisson,
            Nat,
            {"mu": Rplus},
            lambda value, mu: sp.poisson.logpmf(value, mu),
        )
        self.check_logcdf(
            Poisson,
            Nat,
            {"mu": Rplus},
            lambda value, mu: sp.poisson.logcdf(value, mu),
        )
        self.check_selfconsistency_discrete_logcdf(
            Poisson,
            Nat,
            {"mu": Rplus},
        )

    def test_bound_poisson(self):
        NonZeroPoisson = Bound(Poisson, lower=1.0)
        self.check_logp(
            NonZeroPoisson,
            PosNat,
            {"mu": Rplus},
            lambda value, mu: sp.poisson.logpmf(value, mu),
        )

        with Model():
            x = NonZeroPoisson("x", mu=4)
        assert np.isinf(x.logp({"x": 0}))

    def test_constantdist(self):
        self.check_logp(Constant, I, {"c": I}, lambda value, c: np.log(c == value))

    # Too lazy to propagate decimal parameter through the whole chain of deps
    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_zeroinflatedpoisson(self):
        self.checkd(
            ZeroInflatedPoisson,
            Nat,
            {"theta": Rplus, "psi": Unit},
        )
        self.check_selfconsistency_discrete_logcdf(
            ZeroInflatedPoisson,
            Nat,
            {"theta": Rplus, "psi": Unit},
        )

    # Too lazy to propagate decimal parameter through the whole chain of deps
    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_zeroinflatednegativebinomial(self):
        self.checkd(
            ZeroInflatedNegativeBinomial,
            Nat,
            {"mu": Rplusbig, "alpha": Rplusbig, "psi": Unit},
        )
        self.check_selfconsistency_discrete_logcdf(
            ZeroInflatedNegativeBinomial,
            Nat,
            {"mu": Rplusbig, "alpha": Rplusbig, "psi": Unit},
            n_samples=10,
        )

    # Too lazy to propagate decimal parameter through the whole chain of deps
    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_zeroinflatedbinomial(self):
        self.checkd(
            ZeroInflatedBinomial,
            Nat,
            {"n": NatSmall, "p": Unit, "psi": Unit},
        )
        self.check_selfconsistency_discrete_logcdf(
            ZeroInflatedBinomial,
            Nat,
            {"n": NatSmall, "p": Unit, "psi": Unit},
            n_samples=10,
        )

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_mvnormal(self, n):
        self.check_logp(
            MvNormal,
            RealMatrix(5, n),
            {"mu": Vector(R, n), "tau": PdMatrix(n)},
            normal_logpdf_tau,
        )
        self.check_logp(
            MvNormal,
            Vector(R, n),
            {"mu": Vector(R, n), "tau": PdMatrix(n)},
            normal_logpdf_tau,
        )
        self.check_logp(
            MvNormal,
            RealMatrix(5, n),
            {"mu": Vector(R, n), "cov": PdMatrix(n)},
            normal_logpdf_cov,
        )
        self.check_logp(
            MvNormal,
            Vector(R, n),
            {"mu": Vector(R, n), "cov": PdMatrix(n)},
            normal_logpdf_cov,
        )
        self.check_logp(
            MvNormal,
            RealMatrix(5, n),
            {"mu": Vector(R, n), "chol": PdMatrixChol(n)},
            normal_logpdf_chol,
            decimal=select_by_precision(float64=6, float32=-1),
        )
        self.check_logp(
            MvNormal,
            Vector(R, n),
            {"mu": Vector(R, n), "chol": PdMatrixChol(n)},
            normal_logpdf_chol,
            decimal=select_by_precision(float64=6, float32=0),
        )

        def MvNormalUpper(*args, **kwargs):
            return MvNormal(lower=False, *args, **kwargs)

        self.check_logp(
            MvNormalUpper,
            Vector(R, n),
            {"mu": Vector(R, n), "chol": PdMatrixCholUpper(n)},
            normal_logpdf_chol_upper,
            decimal=select_by_precision(float64=6, float32=0),
        )

    @pytest.mark.xfail(
        condition=(theano.config.floatX == "float32"),
        reason="Fails on float32 due to inf issues",
    )
    def test_mvnormal_indef(self):
        cov_val = np.array([[1, 0.5], [0.5, -2]])
        cov = tt.matrix("cov")
        cov.tag.test_value = np.eye(2)
        mu = floatX(np.zeros(2))
        x = tt.vector("x")
        x.tag.test_value = np.zeros(2)
        logp = MvNormal.dist(mu=mu, cov=cov).logp(x)
        f_logp = theano.function([cov, x], logp)
        assert f_logp(cov_val, np.ones(2)) == -np.inf
        dlogp = tt.grad(logp, cov)
        f_dlogp = theano.function([cov, x], dlogp)
        assert not np.all(np.isfinite(f_dlogp(cov_val, np.ones(2))))

        logp = MvNormal.dist(mu=mu, tau=cov).logp(x)
        f_logp = theano.function([cov, x], logp)
        assert f_logp(cov_val, np.ones(2)) == -np.inf
        dlogp = tt.grad(logp, cov)
        f_dlogp = theano.function([cov, x], dlogp)
        assert not np.all(np.isfinite(f_dlogp(cov_val, np.ones(2))))

    def test_mvnormal_init_fail(self):
        with Model():
            with pytest.raises(ValueError):
                x = MvNormal("x", mu=np.zeros(3), shape=3)
            with pytest.raises(ValueError):
                x = MvNormal("x", mu=np.zeros(3), cov=np.eye(3), tau=np.eye(3), shape=3)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_matrixnormal(self, n):
        mat_scale = 1e3  # To reduce logp magnitude
        mean_scale = 0.1
        self.check_logp(
            MatrixNormal,
            RealMatrix(n, n),
            {
                "mu": RealMatrix(n, n) * mean_scale,
                "rowcov": PdMatrix(n) * mat_scale,
                "colcov": PdMatrix(n) * mat_scale,
            },
            matrix_normal_logpdf_cov,
        )
        self.check_logp(
            MatrixNormal,
            RealMatrix(2, n),
            {
                "mu": RealMatrix(2, n) * mean_scale,
                "rowcov": PdMatrix(2) * mat_scale,
                "colcov": PdMatrix(n) * mat_scale,
            },
            matrix_normal_logpdf_cov,
        )
        self.check_logp(
            MatrixNormal,
            RealMatrix(3, n),
            {
                "mu": RealMatrix(3, n) * mean_scale,
                "rowchol": PdMatrixChol(3) * mat_scale,
                "colchol": PdMatrixChol(n) * mat_scale,
            },
            matrix_normal_logpdf_chol,
            decimal=select_by_precision(float64=6, float32=-1),
        )
        self.check_logp(
            MatrixNormal,
            RealMatrix(n, 3),
            {
                "mu": RealMatrix(n, 3) * mean_scale,
                "rowchol": PdMatrixChol(n) * mat_scale,
                "colchol": PdMatrixChol(3) * mat_scale,
            },
            matrix_normal_logpdf_chol,
            decimal=select_by_precision(float64=6, float32=0),
        )

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("m", [3])
    @pytest.mark.parametrize("sigma", [None, 1.0])
    def test_kroneckernormal(self, n, m, sigma):
        np.random.seed(5)
        N = n * m
        covs = [RandomPdMatrix(n), RandomPdMatrix(m)]
        chols = list(map(np.linalg.cholesky, covs))
        evds = list(map(np.linalg.eigh, covs))
        dom = Domain([np.random.randn(N) * 0.1], edges=(None, None), shape=N)
        mu = Domain([np.random.randn(N) * 0.1], edges=(None, None), shape=N)

        std_args = {"mu": mu}
        cov_args = {"covs": covs}
        chol_args = {"chols": chols}
        evd_args = {"evds": evds}
        if sigma is not None and sigma != 0:
            std_args["sigma"] = Domain([sigma], edges=(None, None))
        else:
            for args in [cov_args, chol_args, evd_args]:
                args["sigma"] = sigma

        self.check_logp(
            KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_cov,
            extra_args=cov_args,
            scipy_args=cov_args,
        )
        self.check_logp(
            KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_chol,
            extra_args=chol_args,
            scipy_args=chol_args,
        )
        self.check_logp(
            KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_evd,
            extra_args=evd_args,
            scipy_args=evd_args,
        )

        dom = Domain([np.random.randn(2, N) * 0.1], edges=(None, None), shape=(2, N))

        self.check_logp(
            KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_cov,
            extra_args=cov_args,
            scipy_args=cov_args,
        )
        self.check_logp(
            KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_chol,
            extra_args=chol_args,
            scipy_args=chol_args,
        )
        self.check_logp(
            KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_evd,
            extra_args=evd_args,
            scipy_args=evd_args,
        )

    @pytest.mark.parametrize("n", [1, 2])
    def test_mvt(self, n):
        self.check_logp(
            MvStudentT,
            Vector(R, n),
            {"nu": Rplus, "Sigma": PdMatrix(n), "mu": Vector(R, n)},
            mvt_logpdf,
        )
        self.check_logp(
            MvStudentT,
            RealMatrix(2, n),
            {"nu": Rplus, "Sigma": PdMatrix(n), "mu": Vector(R, n)},
            mvt_logpdf,
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_AR1(self, n):
        self.check_logp(AR1, Vector(R, n), {"k": Unit, "tau_e": Rplus}, AR1_logpdf)

    @pytest.mark.parametrize("n", [2, 3])
    def test_wishart(self, n):
        # This check compares the autodiff gradient to the numdiff gradient.
        # However, due to the strict constraints of the wishart,
        # it is impossible to numerically determine the gradient as a small
        # pertubation breaks the symmetry. Thus disabling. Also, numdifftools was
        # removed in June 2019, so an alternative would be needed.
        #
        # self.checkd(Wishart, PdMatrix(n), {'n': Domain([2, 3, 4, 2000]), 'V': PdMatrix(n)},
        #             checks=[self.check_dlogp])
        pass

    @pytest.mark.parametrize("x,eta,n,lp", LKJ_CASES)
    def test_lkj(self, x, eta, n, lp):
        with Model() as model:
            LKJCorr("lkj", eta=eta, n=n, transform=None)

        pt = {"lkj": x}
        decimals = select_by_precision(float64=6, float32=4)
        assert_almost_equal(model.fastlogp(pt), lp, decimal=decimals, err_msg=str(pt))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_dirichlet(self, n):
        self.check_logp(Dirichlet, Simplex(n), {"a": Vector(Rplus, n)}, dirichlet_logpdf)

    @pytest.mark.parametrize("dist_shape", [1, (2, 1), (1, 2), (2, 4, 3)])
    def test_dirichlet_with_batch_shapes(self, dist_shape):
        a = np.ones(dist_shape)
        with pm.Model() as model:
            d = pm.Dirichlet("a", a=a)

        pymc3_res = d.distribution.logp(d.tag.test_value).eval()
        for idx in np.ndindex(a.shape[:-1]):
            scipy_res = scipy.stats.dirichlet(a[idx]).logpdf(d.tag.test_value[idx])
            assert_almost_equal(pymc3_res[idx], scipy_res)

    def test_dirichlet_shape(self):
        a = tt.as_tensor_variable(np.r_[1, 2])
        with pytest.warns(DeprecationWarning):
            dir_rv = Dirichlet.dist(a)
            assert dir_rv.shape == (2,)

        with pytest.warns(DeprecationWarning), theano.change_flags(compute_test_value="ignore"):
            dir_rv = Dirichlet.dist(tt.vector())

    def test_dirichlet_2D(self):
        self.check_logp(
            Dirichlet,
            MultiSimplex(2, 2),
            {"a": Vector(Vector(Rplus, 2), 2)},
            dirichlet_logpdf,
        )

    @pytest.mark.parametrize("n", [2, 3])
    def test_multinomial(self, n):
        self.check_logp(
            Multinomial, Vector(Nat, n), {"p": Simplex(n), "n": Nat}, multinomial_logpdf
        )

    @pytest.mark.parametrize(
        "p,n",
        [
            [[0.25, 0.25, 0.25, 0.25], 1],
            [[0.3, 0.6, 0.05, 0.05], 2],
            [[0.3, 0.6, 0.05, 0.05], 10],
        ],
    )
    def test_multinomial_mode(self, p, n):
        _p = np.array(p)
        with Model() as model:
            m = Multinomial("m", n, _p, _p.shape)
        assert_allclose(m.distribution.mode.eval().sum(), n)
        _p = np.array([p, p])
        with Model() as model:
            m = Multinomial("m", n, _p, _p.shape)
        assert_allclose(m.distribution.mode.eval().sum(axis=-1), n)

    @pytest.mark.parametrize(
        "p, shape, n",
        [
            [[0.25, 0.25, 0.25, 0.25], 4, 2],
            [[0.25, 0.25, 0.25, 0.25], (1, 4), 3],
            # 3: expect to fail
            # [[.25, .25, .25, .25], (10, 4)],
            [[0.25, 0.25, 0.25, 0.25], (10, 1, 4), 5],
            # 5: expect to fail
            # [[[.25, .25, .25, .25]], (2, 4), [7, 11]],
            [[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]], (2, 4), 13],
            [[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]], (1, 2, 4), [23, 29]],
            [
                [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
                (10, 2, 4),
                [31, 37],
            ],
            [[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]], (2, 4), [17, 19]],
        ],
    )
    def test_multinomial_random(self, p, shape, n):
        p = np.asarray(p)
        with Model() as model:
            m = Multinomial("m", n=n, p=p, shape=shape)
        m.random()

    def test_multinomial_mode_with_shape(self):
        n = [1, 10]
        p = np.asarray([[0.25, 0.25, 0.25, 0.25], [0.26, 0.26, 0.26, 0.22]])
        with Model() as model:
            m = Multinomial("m", n=n, p=p, shape=(2, 4))
        assert_allclose(m.distribution.mode.eval().sum(axis=-1), n)

    def test_multinomial_vec(self):
        vals = np.array([[2, 4, 4], [3, 3, 4]])
        p = np.array([0.2, 0.3, 0.5])
        n = 10

        with Model() as model_single:
            Multinomial("m", n=n, p=p, shape=len(p))

        with Model() as model_many:
            Multinomial("m", n=n, p=p, shape=vals.shape)

        assert_almost_equal(
            scipy.stats.multinomial.logpmf(vals, n, p),
            np.asarray([model_single.fastlogp({"m": val}) for val in vals]),
            decimal=4,
        )

        assert_almost_equal(
            scipy.stats.multinomial.logpmf(vals, n, p),
            model_many.free_RVs[0].logp_elemwise({"m": vals}).squeeze(),
            decimal=4,
        )

        assert_almost_equal(
            sum(model_single.fastlogp({"m": val}) for val in vals),
            model_many.fastlogp({"m": vals}),
            decimal=4,
        )

    def test_multinomial_vec_1d_n(self):
        vals = np.array([[2, 4, 4], [4, 3, 4]])
        p = np.array([0.2, 0.3, 0.5])
        ns = np.array([10, 11])

        with Model() as model:
            Multinomial("m", n=ns, p=p, shape=vals.shape)

        assert_almost_equal(
            sum(multinomial_logpdf(val, n, p) for val, n in zip(vals, ns)),
            model.fastlogp({"m": vals}),
            decimal=4,
        )

    def test_multinomial_vec_1d_n_2d_p(self):
        vals = np.array([[2, 4, 4], [4, 3, 4]])
        ps = np.array([[0.2, 0.3, 0.5], [0.9, 0.09, 0.01]])
        ns = np.array([10, 11])

        with Model() as model:
            Multinomial("m", n=ns, p=ps, shape=vals.shape)

        assert_almost_equal(
            sum(multinomial_logpdf(val, n, p) for val, n, p in zip(vals, ns, ps)),
            model.fastlogp({"m": vals}),
            decimal=4,
        )

    def test_multinomial_vec_2d_p(self):
        vals = np.array([[2, 4, 4], [3, 3, 4]])
        ps = np.array([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
        n = 10

        with Model() as model:
            Multinomial("m", n=n, p=ps, shape=vals.shape)

        assert_almost_equal(
            sum(multinomial_logpdf(val, n, p) for val, p in zip(vals, ps)),
            model.fastlogp({"m": vals}),
            decimal=4,
        )

    def test_batch_multinomial(self):
        n = 10
        vals = np.zeros((4, 5, 3), dtype="int32")
        p = np.zeros_like(vals, dtype=theano.config.floatX)
        inds = np.random.randint(vals.shape[-1], size=vals.shape[:-1])[..., None]
        np.put_along_axis(vals, inds, n, axis=-1)
        np.put_along_axis(p, inds, 1, axis=-1)

        dist = Multinomial.dist(n=n, p=p, shape=vals.shape)
        value = tt.tensor3(dtype="int32")
        value.tag.test_value = np.zeros_like(vals, dtype="int32")
        logp = tt.exp(dist.logp(value))
        f = theano.function(inputs=[value], outputs=logp)
        assert_almost_equal(
            f(vals),
            np.ones(vals.shape[:-1] + (1,)),
            decimal=select_by_precision(float64=6, float32=3),
        )

        sample = dist.random(size=2)
        assert_allclose(sample, np.stack([vals, vals], axis=0))

    @pytest.mark.parametrize("n", [2, 3])
    def test_dirichlet_multinomial(self, n):
        self.check_logp(
            DirichletMultinomial,
            Vector(Nat, n),
            {"a": Vector(Rplus, n), "n": Nat},
            dirichlet_multinomial_logpmf,
        )

    def test_dirichlet_multinomial_matches_beta_binomial(self):
        a, b, n = 2, 1, 5
        ns = np.arange(n + 1)
        ns_dm = np.vstack((ns, n - ns)).T  # covert ns=1 to ns_dm=[1, 4], for all ns...
        bb_logp = pm.BetaBinomial.dist(n=n, alpha=a, beta=b).logp(ns).tag.test_value
        dm_logp = (
            pm.DirichletMultinomial.dist(n=n, a=[a, b], shape=(1, 2)).logp(ns_dm).tag.test_value
        )
        dm_logp = dm_logp.ravel()
        assert_almost_equal(
            dm_logp,
            bb_logp,
            decimal=select_by_precision(float64=6, float32=3),
        )

    @pytest.mark.parametrize(
        "a, n, shape",
        [
            [[0.25, 0.25, 0.25, 0.25], 1, (1, 4)],
            [[0.3, 0.6, 0.05, 0.05], 2, (1, 4)],
            [[0.3, 0.6, 0.05, 0.05], 10, (1, 4)],
            [[0.25, 0.25, 0.25, 0.25], 1, (2, 4)],
            [[0.3, 0.6, 0.05, 0.05], 2, (3, 4)],
            [[[0.25, 0.25, 0.25, 0.25], [0.26, 0.26, 0.26, 0.22]], [1, 10], (2, 4)],
        ],
    )
    def test_dirichlet_multinomial_defaultval(self, a, n, shape):
        a = np.asarray(a)
        with Model() as model:
            m = DirichletMultinomial("m", n=n, a=a, shape=shape)
        assert_allclose(m.distribution._defaultval.eval().sum(axis=-1), n)

    def test_dirichlet_multinomial_vec(self):
        vals = np.array([[2, 4, 4], [3, 3, 4]])
        a = np.array([0.2, 0.3, 0.5])
        n = 10

        with Model() as model_single:
            DirichletMultinomial("m", n=n, a=a, shape=len(a))

        with Model() as model_many:
            DirichletMultinomial("m", n=n, a=a, shape=vals.shape)

        assert_almost_equal(
            np.asarray([dirichlet_multinomial_logpmf(v, n, a) for v in vals]),
            np.asarray([model_single.fastlogp({"m": val}) for val in vals]),
            decimal=4,
        )

        assert_almost_equal(
            np.asarray([dirichlet_multinomial_logpmf(v, n, a) for v in vals]),
            model_many.free_RVs[0].logp_elemwise({"m": vals}).squeeze(),
            decimal=4,
        )

        assert_almost_equal(
            sum(model_single.fastlogp({"m": val}) for val in vals),
            model_many.fastlogp({"m": vals}),
            decimal=4,
        )

    def test_dirichlet_multinomial_vec_1d_n(self):
        vals = np.array([[2, 4, 4], [4, 3, 4]])
        a = np.array([0.2, 0.3, 0.5])
        ns = np.array([10, 11])

        with Model() as model:
            DirichletMultinomial("m", n=ns, a=a, shape=vals.shape)

        assert_almost_equal(
            sum(dirichlet_multinomial_logpmf(val, n, a) for val, n in zip(vals, ns)),
            model.fastlogp({"m": vals}),
            decimal=4,
        )

    def test_dirichlet_multinomial_vec_1d_n_2d_a(self):
        vals = np.array([[2, 4, 4], [4, 3, 4]])
        as_ = np.array([[0.2, 0.3, 0.5], [0.9, 0.09, 0.01]])
        ns = np.array([10, 11])

        with Model() as model:
            DirichletMultinomial("m", n=ns, a=as_, shape=vals.shape)

        assert_almost_equal(
            sum(dirichlet_multinomial_logpmf(val, n, a) for val, n, a in zip(vals, ns, as_)),
            model.fastlogp({"m": vals}),
            decimal=4,
        )

    def test_dirichlet_multinomial_vec_2d_a(self):
        vals = np.array([[2, 4, 4], [3, 3, 4]])
        as_ = np.array([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
        n = 10

        with Model() as model:
            DirichletMultinomial("m", n=n, a=as_, shape=vals.shape)

        assert_almost_equal(
            sum(dirichlet_multinomial_logpmf(val, n, a) for val, a in zip(vals, as_)),
            model.fastlogp({"m": vals}),
            decimal=4,
        )

    def test_batch_dirichlet_multinomial(self):
        # Test that DM can handle a 3d array for `a`

        # Create an almost deterministic DM by setting a to 0.001, everywehere
        # except for one category / dimension which is given the value of 1000
        n = 5
        vals = np.zeros((4, 5, 3), dtype="int32")
        a = np.zeros_like(vals, dtype=theano.config.floatX) + 0.001
        inds = np.random.randint(vals.shape[-1], size=vals.shape[:-1])[..., None]
        np.put_along_axis(vals, inds, n, axis=-1)
        np.put_along_axis(a, inds, 1000, axis=-1)

        dist = DirichletMultinomial.dist(n=n, a=a, shape=vals.shape)

        # Logp should be approx -9.924431e-06
        dist_logp = dist.logp(vals).tag.test_value
        expected_logp = np.full(shape=vals.shape[:-1] + (1,), fill_value=-9.924431e-06)
        assert_almost_equal(
            dist_logp,
            expected_logp,
            decimal=select_by_precision(float64=6, float32=3),
        )

        # Samples should be equal given the almost deterministic DM
        sample = dist.random(size=2)
        assert_allclose(sample, np.stack([vals, vals], axis=0))

    def test_categorical_bounds(self):
        with Model():
            x = Categorical("x", p=np.array([0.2, 0.3, 0.5]))
            assert np.isinf(x.logp({"x": -1}))
            assert np.isinf(x.logp({"x": 3}))

    def test_categorical_valid_p(self):
        with Model():
            x = Categorical("x", p=np.array([-0.2, 0.3, 0.5]))
            assert np.isinf(x.logp({"x": 0}))
            assert np.isinf(x.logp({"x": 1}))
            assert np.isinf(x.logp({"x": 2}))
        with Model():
            # A model where p sums to 1 but contains negative values
            x = Categorical("x", p=np.array([-0.2, 0.7, 0.5]))
            assert np.isinf(x.logp({"x": 0}))
            assert np.isinf(x.logp({"x": 1}))
            assert np.isinf(x.logp({"x": 2}))
        with Model():
            # Hard edge case from #2082
            # Early automatic normalization of p's sum would hide the negative
            # entries if there is a single or pair number of negative values
            # and the rest are zero
            x = Categorical("x", p=np.array([-1, -1, 0, 0]))
            assert np.isinf(x.logp({"x": 0}))
            assert np.isinf(x.logp({"x": 1}))
            assert np.isinf(x.logp({"x": 2}))
            assert np.isinf(x.logp({"x": 3}))

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_categorical(self, n):
        self.check_logp(
            Categorical,
            Domain(range(n), "int64"),
            {"p": Simplex(n)},
            lambda value, p: categorical_logpdf(value, p),
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_orderedlogistic(self, n):
        self.check_logp(
            OrderedLogistic,
            Domain(range(n), "int64"),
            {"eta": R, "cutpoints": Vector(R, n - 1)},
            lambda value, eta, cutpoints: orderedlogistic_logpdf(value, eta, cutpoints),
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_orderedprobit(self, n):
        self.check_logp(
            OrderedProbit,
            Domain(range(n), "int64"),
            {"eta": Runif, "cutpoints": UnitSortedVector(n - 1)},
            lambda value, eta, cutpoints: orderedprobit_logpdf(value, eta, cutpoints),
        )

    def test_densitydist(self):
        def logp(x):
            return -log(2 * 0.5) - abs(x - 0.5) / 0.5

        self.checkd(DensityDist, R, {}, extra_args={"logp": logp})

    def test_get_tau_sigma(self):
        sigma = np.array([2])
        assert_almost_equal(continuous.get_tau_sigma(sigma=sigma), [1.0 / sigma ** 2, sigma])

    @pytest.mark.parametrize(
        "value,mu,sigma,nu,logp",
        [
            (0.5, -50.000, 0.500, 0.500, -99.8068528),
            (1.0, -1.000, 0.001, 0.001, -1992.5922447),
            (2.0, 0.001, 1.000, 1.000, -1.6720416),
            (5.0, 0.500, 2.500, 2.500, -2.4543644),
            (7.5, 2.000, 5.000, 5.000, -2.8259429),
            (15.0, 5.000, 7.500, 7.500, -3.3093854),
            (50.0, 50.000, 10.000, 10.000, -3.6436067),
            (1000.0, 500.000, 10.000, 20.000, -27.8707323),
            (-1.0, 1.0, 20.0, 0.9, -3.91967108),  # Fails in scipy version
            (0.01, 0.01, 100.0, 0.01, -5.5241087),  # Fails in scipy version
            (-1.0, 0.0, 0.1, 0.1, -51.022349),  # Fails in previous pymc3 version
        ],
    )
    def test_ex_gaussian(self, value, mu, sigma, nu, logp):
        """Log probabilities calculated using the dexGAUS function from the R package gamlss.
        See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or http://www.gamlss.org/."""
        with Model() as model:
            ExGaussian("eg", mu=mu, sigma=sigma, nu=nu)
        pt = {"eg": value}
        assert_almost_equal(
            model.fastlogp(pt),
            logp,
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str(pt),
        )

    @pytest.mark.parametrize(
        "value,mu,sigma,nu,logcdf",
        [
            (0.5, -50.000, 0.500, 0.500, 0.0000000),
            (1.0, -1.000, 0.001, 0.001, 0.0000000),
            (2.0, 0.001, 1.000, 1.000, -0.2365674),
            (5.0, 0.500, 2.500, 2.500, -0.2886489),
            (7.5, 2.000, 5.000, 5.000, -0.5655104),
            (15.0, 5.000, 7.500, 7.500, -0.4545255),
            (50.0, 50.000, 10.000, 10.000, -1.433714),
            (1000.0, 500.000, 10.000, 20.000, -1.573708e-11),
            (0.01, 0.01, 100.0, 0.01, -0.69314718),  # Fails in scipy version
            (-0.43402407, 0.0, 0.1, 0.1, -13.59615423),  # Previous 32-bit version failed here
            (-0.72402009, 0.0, 0.1, 0.1, -31.26571842),  # Previous 64-bit version failed here
        ],
    )
    def test_ex_gaussian_cdf(self, value, mu, sigma, nu, logcdf):
        """Log probabilities calculated using the pexGAUS function from the R package gamlss.
        See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or http://www.gamlss.org/."""
        assert_almost_equal(
            ExGaussian.dist(mu=mu, sigma=sigma, nu=nu).logcdf(value).tag.test_value,
            logcdf,
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str((value, mu, sigma, nu, logcdf)),
        )

    def test_ex_gaussian_cdf_outside_edges(self):
        self.check_logcdf(
            ExGaussian,
            R,
            {"mu": R, "sigma": Rplus, "nu": Rplus},
            None,
            skip_paramdomain_inside_edge_test=True,  # Valid values are tested above
        )

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_vonmises(self):
        self.check_logp(
            VonMises,
            R,
            {"mu": Circ, "kappa": Rplus},
            lambda value, mu, kappa: floatX(sp.vonmises.logpdf(value, kappa, loc=mu)),
        )

    def test_gumbel(self):
        def gumbel(value, mu, beta):
            return floatX(sp.gumbel_r.logpdf(value, loc=mu, scale=beta))

        self.check_logp(Gumbel, R, {"mu": R, "beta": Rplusbig}, gumbel)

        def gumbellcdf(value, mu, beta):
            return floatX(sp.gumbel_r.logcdf(value, loc=mu, scale=beta))

        self.check_logcdf(Gumbel, R, {"mu": R, "beta": Rplusbig}, gumbellcdf)

    def test_logistic(self):
        self.check_logp(
            Logistic,
            R,
            {"mu": R, "s": Rplus},
            lambda value, mu, s: sp.logistic.logpdf(value, mu, s),
            decimal=select_by_precision(float64=6, float32=1),
        )
        self.check_logcdf(
            Logistic,
            R,
            {"mu": R, "s": Rplus},
            lambda value, mu, s: sp.logistic.logcdf(value, mu, s),
            decimal=select_by_precision(float64=6, float32=1),
        )

    def test_logitnormal(self):
        self.check_logp(
            LogitNormal,
            Unit,
            {"mu": R, "sigma": Rplus},
            lambda value, mu, sigma: (
                sp.norm.logpdf(logit(value), mu, sigma) - (np.log(value) + np.log1p(-value))
            ),
            decimal=select_by_precision(float64=6, float32=1),
        )

    def test_multidimensional_beta_construction(self):
        with Model():
            Beta("beta", alpha=1.0, beta=1.0, shape=(10, 20))

    def test_rice(self):
        self.check_logp(
            Rice,
            Rplus,
            {"nu": Rplus, "sigma": Rplusbig},
            lambda value, nu, sigma: sp.rice.logpdf(value, b=nu / sigma, loc=0, scale=sigma),
        )
        self.check_logp(
            Rice,
            Rplus,
            {"b": Rplus, "sigma": Rplusbig},
            lambda value, b, sigma: sp.rice.logpdf(value, b=b, loc=0, scale=sigma),
        )

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_moyal(self):
        self.check_logp(
            Moyal,
            R,
            {"mu": R, "sigma": Rplusbig},
            lambda value, mu, sigma: floatX(sp.moyal.logpdf(value, mu, sigma)),
        )
        self.check_logcdf(
            Moyal,
            R,
            {"mu": R, "sigma": Rplusbig},
            lambda value, mu, sigma: floatX(sp.moyal.logcdf(value, mu, sigma)),
        )

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_interpolated(self):
        for mu in R.vals:
            for sigma in Rplus.vals:
                # pylint: disable=cell-var-from-loop
                xmin = mu - 5 * sigma
                xmax = mu + 5 * sigma

                class TestedInterpolated(Interpolated):
                    def __init__(self, **kwargs):
                        x_points = np.linspace(xmin, xmax, 100000)
                        pdf_points = sp.norm.pdf(x_points, loc=mu, scale=sigma)
                        super().__init__(x_points=x_points, pdf_points=pdf_points, **kwargs)

                def ref_pdf(value):
                    return np.where(
                        np.logical_and(value >= xmin, value <= xmax),
                        sp.norm.logpdf(value, mu, sigma),
                        -np.inf * np.ones(value.shape),
                    )

                self.check_logp(TestedInterpolated, R, {}, ref_pdf)


def test_bound():
    np.random.seed(42)
    UnboundNormal = Bound(Normal)
    dist = UnboundNormal.dist(mu=0, sigma=1)
    assert dist.transform is None
    assert dist.default() == 0.0
    assert isinstance(dist.random(), np.ndarray)

    LowerNormal = Bound(Normal, lower=1)
    dist = LowerNormal.dist(mu=0, sigma=1)
    assert dist.logp(0).eval() == -np.inf
    assert dist.default() > 1
    assert dist.transform is not None
    assert np.all(dist.random() > 1)

    UpperNormal = Bound(Normal, upper=-1)
    dist = UpperNormal.dist(mu=0, sigma=1)
    assert dist.logp(-0.5).eval() == -np.inf
    assert dist.default() < -1
    assert dist.transform is not None
    assert np.all(dist.random() < -1)

    ArrayNormal = Bound(Normal, lower=[1, 2], upper=[2, 3])
    dist = ArrayNormal.dist(mu=0, sigma=1, shape=2)
    assert_equal(dist.logp([0.5, 3.5]).eval(), -np.array([np.inf, np.inf]))
    assert_equal(dist.default(), np.array([1.5, 2.5]))
    assert dist.transform is not None
    with pytest.raises(ValueError) as err:
        dist.random()
    err.match("Drawing samples from distributions with array-valued")

    with Model():
        a = ArrayNormal("c", shape=2)
        assert_equal(a.tag.test_value, np.array([1.5, 2.5]))

    lower = tt.vector("lower")
    lower.tag.test_value = np.array([1, 2]).astype(theano.config.floatX)
    upper = 3
    ArrayNormal = Bound(Normal, lower=lower, upper=upper)
    dist = ArrayNormal.dist(mu=0, sigma=1, shape=2)
    logp = dist.logp([0.5, 3.5]).eval({lower: lower.tag.test_value})
    assert_equal(logp, -np.array([np.inf, np.inf]))
    assert_equal(dist.default(), np.array([2, 2.5]))
    assert dist.transform is not None

    with Model():
        a = ArrayNormal("c", shape=2)
        assert_equal(a.tag.test_value, np.array([2, 2.5]))

    rand = Bound(Binomial, lower=10).dist(n=20, p=0.3).random()
    assert rand.dtype in [np.int16, np.int32, np.int64]
    assert rand >= 10

    rand = Bound(Binomial, upper=10).dist(n=20, p=0.8).random()
    assert rand.dtype in [np.int16, np.int32, np.int64]
    assert rand <= 10

    rand = Bound(Binomial, lower=5, upper=8).dist(n=10, p=0.6).random()
    assert rand.dtype in [np.int16, np.int32, np.int64]
    assert rand >= 5 and rand <= 8

    with Model():
        BoundPoisson = Bound(Poisson, upper=6)
        BoundPoisson(name="y", mu=1)

    with Model():
        BoundNormalNamedArgs = Bound(Normal, upper=6)("y", mu=2.0, sd=1.0)
        BoundNormalPositionalArgs = Bound(Normal, upper=6)("x", 2.0, 1.0)

    with Model():
        BoundPoissonNamedArgs = Bound(Poisson, upper=6)("y", mu=2.0)
        BoundPoissonPositionalArgs = Bound(Poisson, upper=6)("x", 2.0)


class TestStrAndLatexRepr:
    def setup_class(self):
        # True parameter values
        alpha, sigma = 1, 1
        beta = [1, 2.5]

        # Size of dataset
        size = 100

        # Predictor variable
        X = np.random.normal(size=(size, 2)).dot(np.array([[1, 0], [0, 0.2]]))

        # Simulate outcome variable
        Y = alpha + X.dot(beta) + np.random.randn(size) * sigma
        with Model() as self.model:
            # Priors for unknown model parameters
            alpha = Normal("alpha", mu=0, sigma=10)
            b = Normal("beta", mu=0, sigma=10, shape=(2,), observed=beta)
            sigma = HalfNormal("sigma", sigma=1)

            # Test Cholesky parameterization
            Z = MvNormal("Z", mu=np.zeros(2), chol=np.eye(2), shape=(2,))

            # NegativeBinomial representations to test issue 4186
            nb1 = pm.NegativeBinomial(
                "nb_with_mu_alpha", mu=pm.Normal("nbmu"), alpha=pm.Gamma("nbalpha", mu=6, sigma=1)
            )
            nb2 = pm.NegativeBinomial("nb_with_p_n", p=pm.Uniform("nbp"), n=10)

            # Expected value of outcome
            mu = Deterministic("mu", floatX(alpha + tt.dot(X, b)))

            # add a bounded variable as well
            bound_var = Bound(Normal, lower=1.0)("bound_var", mu=0, sigma=10)

            # KroneckerNormal
            n, m = 3, 4
            covs = [np.eye(n), np.eye(m)]
            kron_normal = KroneckerNormal("kron_normal", mu=np.zeros(n * m), covs=covs, shape=n * m)

            # MatrixNormal
            matrix_normal = MatrixNormal(
                "mat_normal",
                mu=np.random.normal(size=n),
                rowcov=np.eye(n),
                colchol=np.linalg.cholesky(np.eye(n)),
                shape=(n, n),
            )

            # DirichletMultinomial
            dm = DirichletMultinomial("dm", n=5, a=[1, 1, 1], shape=(2, 3))

            # Likelihood (sampling distribution) of observations
            Y_obs = Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

        self.distributions = [alpha, sigma, mu, b, Z, nb1, nb2, Y_obs, bound_var]
        self.expected = {
            "latex": (
                r"$\text{alpha} \sim \text{Normal}$",
                r"$\text{sigma} \sim \text{HalfNormal}$",
                r"$\text{mu} \sim \text{Deterministic}$",
                r"$\text{beta} \sim \text{Normal}$",
                r"$\text{Z} \sim \text{MvNormal}$",
                r"$\text{nb_with_mu_alpha} \sim \text{NegativeBinomial}$",
                r"$\text{nb_with_p_n} \sim \text{NegativeBinomial}$",
                r"$\text{Y_obs} \sim \text{Normal}$",
                r"$\text{bound_var} \sim \text{Bound}$ -- \text{Normal}$",
                r"$\text{kron_normal} \sim \text{KroneckerNormal}$",
                r"$\text{mat_normal} \sim \text{MatrixNormal}$",
                r"$\text{dm} \sim \text{DirichletMultinomial}$",
            ),
            "plain": (
                r"alpha ~ Normal",
                r"sigma ~ HalfNormal",
                r"mu ~ Deterministic",
                r"beta ~ Normal",
                r"Z ~ MvNormal",
                r"nb_with_mu_alpha ~ NegativeBinomial",
                r"nb_with_p_n ~ NegativeBinomial",
                r"Y_obs ~ Normal",
                r"bound_var ~ Bound-Normal",
                r"kron_normal ~ KroneckerNormal",
                r"mat_normal ~ MatrixNormal",
                r"dm ~ DirichletMultinomial",
            ),
            "latex_with_params": (
                r"$\text{alpha} \sim \text{Normal}(\mathit{mu}=0.0,~\mathit{sigma}=10.0)$",
                r"$\text{sigma} \sim \text{HalfNormal}(\mathit{sigma}=1.0)$",
                r"$\text{mu} \sim \text{Deterministic}(\text{alpha},~\text{Constant},~\text{beta})$",
                r"$\text{beta} \sim \text{Normal}(\mathit{mu}=0.0,~\mathit{sigma}=10.0)$",
                r"$\text{Z} \sim \text{MvNormal}(\mathit{mu}=array,~\mathit{chol_cov}=array)$",
                r"$\text{nb_with_mu_alpha} \sim \text{NegativeBinomial}(\mathit{mu}=\text{nbmu},~\mathit{alpha}=\text{nbalpha})$",
                r"$\text{nb_with_p_n} \sim \text{NegativeBinomial}(\mathit{p}=\text{nbp},~\mathit{n}=10)$",
                r"$\text{Y_obs} \sim \text{Normal}(\mathit{mu}=\text{mu},~\mathit{sigma}=f(\text{sigma}))$",
                r"$\text{bound_var} \sim \text{Bound}(\mathit{lower}=1.0,~\mathit{upper}=\text{None})$ -- \text{Normal}(\mathit{mu}=0.0,~\mathit{sigma}=10.0)$",
                r"$\text{kron_normal} \sim \text{KroneckerNormal}(\mathit{mu}=array)$",
                r"$\text{mat_normal} \sim \text{MatrixNormal}(\mathit{mu}=array,~\mathit{rowcov}=array,~\mathit{colchol_cov}=array)$",
                r"$\text{dm} \sim \text{DirichletMultinomial}(\mathit{n}=5,~\mathit{a}=array)$",
            ),
            "plain_with_params": (
                r"alpha ~ Normal(mu=0.0, sigma=10.0)",
                r"sigma ~ HalfNormal(sigma=1.0)",
                r"mu ~ Deterministic(alpha, Constant, beta)",
                r"beta ~ Normal(mu=0.0, sigma=10.0)",
                r"Z ~ MvNormal(mu=array, chol_cov=array)",
                r"nb_with_mu_alpha ~ NegativeBinomial(mu=nbmu, alpha=nbalpha)",
                r"nb_with_p_n ~ NegativeBinomial(p=nbp, n=10)",
                r"Y_obs ~ Normal(mu=mu, sigma=f(sigma))",
                r"bound_var ~ Bound(lower=1.0, upper=None)-Normal(mu=0.0, sigma=10.0)",
                r"kron_normal ~ KroneckerNormal(mu=array)",
                r"mat_normal ~ MatrixNormal(mu=array, rowcov=array, colchol_cov=array)",
                r"dm∼DirichletMultinomial(n=5, a=array)",
            ),
        }

    def test__repr_latex_(self):
        for distribution, tex in zip(self.distributions, self.expected["latex_with_params"]):
            assert distribution._repr_latex_() == tex

        model_tex = self.model._repr_latex_()

        # make sure each variable is in the model
        for tex in self.expected["latex"]:
            for segment in tex.strip("$").split(r"\sim"):
                assert segment in model_tex

    def test___latex__(self):
        for distribution, tex in zip(self.distributions, self.expected["latex_with_params"]):
            assert distribution._repr_latex_() == distribution.__latex__()
        assert self.model._repr_latex_() == self.model.__latex__()

    def test___str__(self):
        for distribution, str_repr in zip(self.distributions, self.expected["plain"]):
            assert distribution.__str__() == str_repr

        model_str = self.model.__str__()
        for str_repr in self.expected["plain"]:
            assert str_repr in model_str

    def test_str(self):
        for distribution, str_repr in zip(self.distributions, self.expected["plain"]):
            assert str(distribution) == str_repr

        model_str = str(self.model)
        for str_repr in self.expected["plain"]:
            assert str_repr in model_str


def test_discrete_trafo():
    with pytest.raises(ValueError) as err:
        Binomial.dist(n=5, p=0.5, transform="log")
    err.match("Transformations for discrete distributions")
    with Model():
        with pytest.raises(ValueError) as err:
            Binomial("a", n=5, p=0.5, transform="log")
        err.match("Transformations for discrete distributions")


@pytest.mark.parametrize("shape", [tuple(), (1,), (3, 1), (3, 2)], ids=str)
def test_orderedlogistic_dimensions(shape):
    # Test for issue #3535
    loge = np.log10(np.exp(1))
    size = 7
    p = np.ones(shape + (10,)) / 10
    cutpoints = np.tile(logit(np.linspace(0, 1, 11)[1:-1]), shape + (1,))
    obs = np.random.randint(0, 1, size=(size,) + shape)
    with Model():
        ol = OrderedLogistic(
            "ol", eta=np.zeros(shape), cutpoints=cutpoints, shape=shape, observed=obs
        )
        c = Categorical("c", p=p, shape=shape, observed=obs)
    ologp = ol.logp({"ol": 1}) * loge
    clogp = c.logp({"c": 1}) * loge
    expected = -np.prod((size,) + shape)

    assert c.distribution.p.ndim == (len(shape) + 1)
    assert np.allclose(clogp, expected)
    assert ol.distribution.p.ndim == (len(shape) + 1)
    assert np.allclose(ologp, expected)


class TestBugfixes:
    @pytest.mark.parametrize(
        "dist_cls,kwargs", [(MvNormal, dict(mu=0)), (MvStudentT, dict(mu=0, nu=2))]
    )
    @pytest.mark.parametrize("dims", [1, 2, 4])
    def test_issue_3051(self, dims, dist_cls, kwargs):
        d = dist_cls.dist(**kwargs, cov=np.eye(dims), shape=(dims,))

        X = np.random.normal(size=(20, dims))
        actual_t = d.logp(X)
        assert isinstance(actual_t, tt.TensorVariable)
        actual_a = actual_t.eval()
        assert isinstance(actual_a, np.ndarray)
        assert actual_a.shape == (X.shape[0],)
        pass


def test_serialize_density_dist():
    def func(x):
        return -2 * (x ** 2).sum()

    with pm.Model():
        pm.Normal("x")
        y = pm.DensityDist("y", func)
        pm.sample(draws=5, tune=1, mp_ctx="spawn", return_inferencedata=False)

    import pickle

    pickle.loads(pickle.dumps(y))
