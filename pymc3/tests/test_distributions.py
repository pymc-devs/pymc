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
import functools
import itertools
import sys

import aesara
import aesara.tensor as at
import numpy as np
import numpy.random as nr

try:
    from polyagamma import polyagamma_cdf, polyagamma_pdf

    _polyagamma_not_installed = False
except ImportError:  # pragma: no cover

    _polyagamma_not_installed = True

    def polyagamma_pdf(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")

    def polyagamma_cdf(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")


import pytest
import scipy.stats
import scipy.stats.distributions as sp

from aesara.compile.mode import Mode
from aesara.graph.basic import ancestors
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable
from numpy import array, inf, log
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from scipy import integrate
from scipy.special import erf, logit

import pymc3 as pm

from pymc3.aesaraf import floatX, intX
from pymc3.distributions import (
    AR1,
    CAR,
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
    Distribution,
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
    LogNormal,
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
    Wishart,
    ZeroInflatedBinomial,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
    continuous,
    logcdf,
    logp,
    logpt,
    logpt_sum,
)
from pymc3.math import kronecker
from pymc3.model import Deterministic, Model, Point, Potential
from pymc3.tests.helpers import select_by_precision
from pymc3.vartypes import continuous_types


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
            avals = avals.astype(aesara.config.floatX)
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
    all_vals = [zip(names, val) for val in itertools.product(*(d.vals for d in domains))]
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
        param_vars = {}
        for v, dom in vardomains.items():
            v_at = aesara.shared(np.asarray(dom.vals[0]))
            v_at.name = v
            param_vars[v] = v_at
        param_vars.update(extra_args)
        distfam(
            "value",
            **param_vars,
            transform=None,
        )
    return m, param_vars


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
        self.vals = list(itertools.product(*(d.vals for d in domains)))
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


def kron_normal_logpdf_cov(value, mu, covs, sigma, size=None):
    cov = kronecker(*covs).eval()
    if sigma is not None:
        cov += sigma ** 2 * np.eye(*cov.shape)
    return scipy.stats.multivariate_normal.logpdf(value, mu, cov).sum()


def kron_normal_logpdf_chol(value, mu, chols, sigma, size=None):
    covs = [np.dot(chol, chol.T) for chol in chols]
    return kron_normal_logpdf_cov(value, mu, covs, sigma=sigma)


def kron_normal_logpdf_evd(value, mu, evds, sigma, size=None):
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
        np.log(np.power(q, np.power(value, beta)) - np.power(q, np.power(value + 1, beta)))
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
    logp_mvt = norm - logdet - (nu + d) / 2.0 * np.log1p((trafo * trafo).sum(-1) / nu)
    return logp_mvt.sum()


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


def test_hierarchical_logpt():
    """Make sure there are no random variables in a model's log-likelihood graph."""
    with pm.Model() as m:
        x = pm.Uniform("x", lower=0, upper=1)
        y = pm.Uniform("y", lower=0, upper=x)

    logpt_ancestors = list(ancestors([m.logpt]))
    ops = {a.owner.op for a in logpt_ancestors if a.owner}
    assert len(ops) > 0
    assert not any(isinstance(o, RandomVariable) for o in ops)
    assert x.tag.value_var in logpt_ancestors
    assert y.tag.value_var in logpt_ancestors


def test_hierarchical_obs_logpt():
    obs = np.array([0.5, 0.4, 5, 2])

    with pm.Model() as model:
        x = pm.Uniform("x", 0, 1, observed=obs)
        pm.Uniform("y", x, 2, observed=obs)

    logpt_ancestors = list(ancestors([model.logpt]))
    ops = {a.owner.op for a in logpt_ancestors if a.owner}
    assert len(ops) > 0
    assert not any(isinstance(o, RandomVariable) for o in ops)


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

        model, param_vars = build_model(pymc3_dist, domain, paramdomains, extra_args)
        logp_pymc3 = model.fastlogp_nojac

        domains = paramdomains.copy()
        domains["value"] = domain
        for pt in product(domains, n_samples=n_samples):
            pt = dict(pt)
            pt_d = self._model_input_dict(model, param_vars, pt)
            pt_logp = Point(pt_d, model=model)
            pt_ref = Point(pt, filter_model_vars=False, model=model)
            assert_almost_equal(
                logp_pymc3(pt_logp),
                logp_reference(pt_ref),
                decimal=decimal,
                err_msg=str(pt),
            )

    def _model_input_dict(self, model, param_vars, pt):
        """Create a dict with only the necessary, transformed logp inputs."""
        pt_d = {}
        for k, v in pt.items():
            rv_var = model.named_vars.get(k)
            nv = param_vars.get(k, rv_var)
            nv = getattr(nv.tag, "value_var", nv)

            transform = getattr(nv.tag, "transform", None)
            if transform:
                # todo: the compiled graph behind this should be cached and
                # reused (if it isn't already).
                v = transform.forward(rv_var, v).eval()

            if nv.name in param_vars:
                # update the shared parameter variables in `param_vars`
                param_vars[nv.name].set_value(v)
            else:
                # create an argument entry for the (potentially
                # transformed) "value" variable
                pt_d[nv.name] = v

        return pt_d

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

            model, param_vars = build_model(pymc3_dist, domain, paramdomains)
            pymc3_logcdf = model.fastfn(logpt(model["value"], cdf=True))

            if decimal is None:
                decimal = select_by_precision(float64=6, float32=3)

            for pt in product(domains, n_samples=n_samples):
                params = dict(pt)
                scipy_eval = scipy_logcdf(**params)

                value = params.pop("value")
                # Update shared parameter variables in pymc3_logcdf function
                for param_name, param_value in params.items():
                    param_vars[param_name].set_value(param_value)
                pymc3_eval = pymc3_logcdf({"value": value})

                params["value"] = value  # for displaying in err_msg
                assert_almost_equal(
                    pymc3_eval,
                    scipy_eval,
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
                        # We need to remove `Assert`s introduced by checks like
                        # `assert_negative_support` and disable test values;
                        # otherwise, we won't be able to create the
                        # `RandomVariable`
                        with aesara.config.change_flags(compute_test_value="off"):
                            invalid_dist = pymc3_dist.dist(**test_params)
                        with aesara.config.change_flags(mode=Mode("py")):
                            assert_equal(
                                logcdf(invalid_dist, valid_value).eval(),
                                -np.inf,
                                err_msg=str(test_params),
                            )

        # Test that values below domain edge evaluate to -np.inf
        if np.isfinite(domain.lower):
            below_domain = domain.lower - 1
            with aesara.config.change_flags(mode=Mode("py")):
                assert_equal(
                    logcdf(valid_dist, below_domain).eval(),
                    -np.inf,
                    err_msg=str(below_domain),
                )

        # Test that values above domain edge evaluate to 0
        if domain not in nat_domains and np.isfinite(domain.upper):
            above_domain = domain.upper + 1
            with aesara.config.change_flags(mode=Mode("py")):
                assert_equal(
                    logcdf(valid_dist, above_domain).eval(),
                    0,
                    err_msg=str(above_domain),
                )

        # Test that method works with multiple values or raises informative TypeError
        valid_dist = pymc3_dist.dist(**valid_params, size=2)
        with aesara.config.change_flags(mode=Mode("py")):
            try:
                logcdf(valid_dist, np.array([valid_value, valid_value])).eval()
            except TypeError as err:
                assert str(err).endswith(
                    "logcdf expects a scalar value but received a 1-dimensional object."
                )

    def check_selfconsistency_discrete_logcdf(
        self,
        distribution,
        domain,
        paramdomains,
        decimal=None,
        n_samples=100,
    ):
        """
        Check that logcdf of discrete distributions matches sum of logps up to value
        """
        # This test only works for scalar random variables
        assert distribution.rv_op.ndim_supp == 0

        domains = paramdomains.copy()
        domains["value"] = domain
        if decimal is None:
            decimal = select_by_precision(float64=6, float32=3)

        model, param_vars = build_model(distribution, domain, paramdomains)
        dist_logcdf = model.fastfn(logpt(model["value"], cdf=True))
        dist_logp = model.fastfn(logpt(model["value"]))

        for pt in product(domains, n_samples=n_samples):
            params = dict(pt)
            value = params.pop("value")
            values = np.arange(domain.lower, value + 1)

            # Update shared parameter variables in logp/logcdf function
            for param_name, param_value in params.items():
                param_vars[param_name].set_value(param_value)

            with aesara.config.change_flags(mode=Mode("py")):
                assert_almost_equal(
                    dist_logcdf({"value": value}),
                    scipy.special.logsumexp([dist_logp({"value": value}) for value in values]),
                    decimal=decimal,
                    err_msg=str(pt),
                )

    def check_int_to_1(self, model, value, domain, paramdomains, n_samples=10):
        pdf = model.fastfn(exp(model.logpt))
        for pt in product(paramdomains, n_samples=n_samples):
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
        with aesara.config.change_flags(mode=Mode("py")):
            assert logp(invalid_dist, np.array(0.5)).eval() == -np.inf
            assert logcdf(invalid_dist, np.array(2.0)).eval() == -np.inf

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

        # Custom logp/logcdf check for values outside of domain
        valid_dist = Triangular.dist(lower=0, upper=1, c=0.9, size=2)
        with aesara.config.change_flags(mode=Mode("py")):
            assert np.all(logp(valid_dist, np.array([-1, 2])).eval() == -np.inf)
            assert np.all(logcdf(valid_dist, np.array([-1, 2])).eval() == [-np.inf, 0])

        # Custom logp / logcdf check for invalid parameters
        invalid_dist = Triangular.dist(lower=1, upper=0, c=0.1)
        with aesara.config.change_flags(mode=Mode("py")):
            assert logp(invalid_dist, 0.5).eval() == -np.inf
            assert logcdf(invalid_dist, 2).eval() == -np.inf

        invalid_dist = Triangular.dist(lower=0, upper=1, c=2.0)
        with aesara.config.change_flags(mode=Mode("py")):
            assert logp(invalid_dist, 0.5).eval() == -np.inf
            assert logcdf(invalid_dist, 2).eval() == -np.inf

    @pytest.mark.skipif(
        condition=_polyagamma_not_installed,
        reason="`polyagamma package is not available/installed.",
    )
    def test_polyagamma(self):
        self.check_logp(
            pm.PolyaGamma,
            Rplus,
            {"h": Rplus, "z": R},
            lambda value, h, z: polyagamma_pdf(value, h, z, return_log=True),
            decimal=select_by_precision(float64=6, float32=-1),
        )
        self.check_logcdf(
            pm.PolyaGamma,
            Rplus,
            {"h": Rplus, "z": R},
            lambda value, h, z: polyagamma_cdf(value, h, z, return_log=True),
            decimal=select_by_precision(float64=6, float32=-1),
        )

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
        with aesara.config.change_flags(mode=Mode("py")):
            assert logp(invalid_dist, 0.5).eval() == -np.inf
            assert logcdf(invalid_dist, 2).eval() == -np.inf

    def test_flat(self):
        self.check_logp(Flat, Runif, {}, lambda value: 0)
        with Model():
            x = Flat("a")
            assert_allclose(x.tag.test_value, 0)
        self.check_logcdf(Flat, R, {}, lambda value: np.log(0.5))
        # Check infinite cases individually.
        assert 0.0 == logcdf(Flat.dist(), np.inf).eval()
        assert -np.inf == logcdf(Flat.dist(), -np.inf).eval()

    def test_half_flat(self):
        self.check_logp(HalfFlat, Rplus, {}, lambda value: 0)
        with Model():
            x = HalfFlat("a", size=2)
            assert_allclose(x.tag.test_value, 1)
            assert x.tag.test_value.shape == (2,)
        self.check_logcdf(HalfFlat, Rplus, {}, lambda value: -np.inf)
        # Check infinite cases individually.
        assert 0.0 == logcdf(HalfFlat.dist(), np.inf).eval()
        assert -np.inf == logcdf(HalfFlat.dist(), -np.inf).eval()

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
            decimal=select_by_precision(float64=6, float32=1),
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

        self.check_logp(
            TruncatedNormal,
            R,
            {"mu": R, "sigma": Rplusbig, "upper": Rplusbig},
            functools.partial(scipy_logp, lower=-np.inf),
            decimal=select_by_precision(float64=6, float32=1),
        )

        self.check_logp(
            TruncatedNormal,
            R,
            {"mu": R, "sigma": Rplusbig, "lower": -Rplusbig},
            functools.partial(scipy_logp, upper=np.inf),
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

    def test_chisquared_logp(self):
        self.check_logp(
            ChiSquared,
            Rplus,
            {"nu": Rplus},
            lambda value, nu: sp.chi2.logpdf(value, df=nu),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_chisquared_logcdf(self):
        self.check_logcdf(
            ChiSquared,
            Rplus,
            {"nu": Rplus},
            lambda value, nu: sp.chi2.logcdf(value, df=nu),
        )

    def test_wald_logp(self):
        self.check_logp(
            Wald,
            Rplus,
            {"mu": Rplus, "alpha": Rplus},
            lambda value, mu, alpha: sp.invgauss.logpdf(value, mu=mu, loc=alpha),
            decimal=select_by_precision(float64=6, float32=1),
        )

    @pytest.mark.xfail(
        condition=(aesara.config.floatX == "float32"),
        reason="Poor CDF in SciPy. See scipy/scipy#869 for details.",
    )
    def test_wald_logcdf(self):
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
    def test_wald_logp_custom_points(self, value, mu, lam, phi, alpha, logp):
        # Log probabilities calculated using the dIG function from the R package gamlss.
        # See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or
        # http://www.gamlss.org/.
        with Model() as model:
            Wald("wald", mu=mu, lam=lam, phi=phi, alpha=alpha, transform=None)
        pt = {"wald": value}
        decimals = select_by_precision(float64=6, float32=1)
        assert_almost_equal(model.fastlogp(pt), logp, decimal=decimals, err_msg=str(pt))

    def test_beta_logp(self):
        self.check_logp(
            Beta,
            Unit,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: sp.beta.logpdf(value, alpha, beta),
        )
        self.check_logp(
            Beta,
            Unit,
            {"mu": Unit, "sigma": Rplus},
            beta_mu_sigma,
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_beta_logcdf(self):
        self.check_logcdf(
            Beta,
            Unit,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: sp.beta.logcdf(value, alpha, beta),
        )

    def test_kumaraswamy(self):
        # Scipy does not have a built-in Kumaraswamy
        def scipy_log_pdf(value, a, b):
            return (
                np.log(a) + np.log(b) + (a - 1) * np.log(value) + (b - 1) * np.log(1 - value ** a)
            )

        def scipy_log_cdf(value, a, b):
            return pm.math.log1mexp_numpy(b * np.log1p(-(value ** a)), negative_input=True)

        self.check_logp(
            Kumaraswamy,
            Unit,
            {"a": Rplus, "b": Rplus},
            scipy_log_pdf,
        )
        self.check_logcdf(
            Kumaraswamy,
            Unit,
            {"a": Rplus, "b": Rplus},
            scipy_log_cdf,
        )

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
        )
        self.check_logcdf(
            NegativeBinomial,
            Nat,
            {"p": Unit, "n": Rplus},
            lambda value, p, n: sp.nbinom.logcdf(value, n, p),
        )
        self.check_selfconsistency_discrete_logcdf(
            NegativeBinomial,
            Nat,
            {"mu": Rplus, "alpha": Rplus},
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
            decimal=select_by_precision(float64=6, float32=2),
        )

    def test_lognormal(self):
        self.check_logp(
            LogNormal,
            Rplus,
            {"mu": R, "tau": Rplusbig},
            lambda value, mu, tau: floatX(sp.lognorm.logpdf(value, tau ** -0.5, 0, np.exp(mu))),
        )
        self.check_logp(
            LogNormal,
            Rplus,
            {"mu": R, "sigma": Rplusbig},
            lambda value, mu, sigma: floatX(sp.lognorm.logpdf(value, sigma, 0, np.exp(mu))),
        )
        self.check_logcdf(
            LogNormal,
            Rplus,
            {"mu": R, "tau": Rplusbig},
            lambda value, mu, tau: sp.lognorm.logcdf(value, tau ** -0.5, 0, np.exp(mu)),
        )
        self.check_logcdf(
            LogNormal,
            Rplus,
            {"mu": R, "sigma": Rplusbig},
            lambda value, mu, sigma: sp.lognorm.logcdf(value, sigma, 0, np.exp(mu)),
        )

    def test_studentt_logp(self):
        self.check_logp(
            StudentT,
            R,
            {"nu": Rplus, "mu": R, "lam": Rplus},
            lambda value, nu, mu, lam: sp.t.logpdf(value, nu, mu, lam ** -0.5),
        )
        self.check_logp(
            StudentT,
            R,
            {"nu": Rplus, "mu": R, "sigma": Rplus},
            lambda value, nu, mu, sigma: sp.t.logpdf(value, nu, mu, sigma),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_studentt_logcdf(self):
        self.check_logcdf(
            StudentT,
            R,
            {"nu": Rplus, "mu": R, "lam": Rplus},
            lambda value, nu, mu, lam: sp.t.logcdf(value, nu, mu, lam ** -0.5),
        )
        self.check_logcdf(
            StudentT,
            R,
            {"nu": Rplus, "mu": R, "sigma": Rplus},
            lambda value, nu, mu, sigma: sp.t.logcdf(value, nu, mu, sigma),
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

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_gamma_logcdf(self):
        self.check_logcdf(
            Gamma,
            Rplus,
            {"alpha": Rplusbig, "beta": Rplusbig},
            lambda value, alpha, beta: sp.gamma.logcdf(value, alpha, scale=1.0 / beta),
        )

    def test_inverse_gamma_logp(self):
        self.check_logp(
            InverseGamma,
            Rplus,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: sp.invgamma.logpdf(value, alpha, scale=beta),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_inverse_gamma_logcdf(self):
        self.check_logcdf(
            InverseGamma,
            Rplus,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: sp.invgamma.logcdf(value, alpha, scale=beta),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
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
            decimal=select_by_precision(float64=4, float32=3),
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

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_weibull_logp(self):
        self.check_logp(
            Weibull,
            Rplus,
            {"alpha": Rplusbig, "beta": Rplusbig},
            lambda value, alpha, beta: sp.exponweib.logpdf(value, 1, alpha, scale=beta),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to inf issues",
    )
    def test_weibull_logcdf(self):
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
        )
        self.check_selfconsistency_discrete_logcdf(
            Binomial,
            Nat,
            {"n": NatSmall, "p": Unit},
        )

    @pytest.mark.xfail(reason="checkd tests has not been refactored")
    @pytest.mark.skipif(condition=(aesara.config.floatX == "float32"), reason="Fails on float32")
    def test_beta_binomial_distribution(self):
        self.checkd(
            BetaBinomial,
            Nat,
            {"alpha": Rplus, "beta": Rplus, "n": NatSmall},
        )

    def test_beta_binomial(self):
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
            {"p": Unit},
            lambda value, p: sp.bernoulli.logpmf(value, p),
        )
        self.check_logp(
            Bernoulli,
            Bool,
            {"logit_p": R},
            lambda value, logit_p: sp.bernoulli.logpmf(value, scipy.special.expit(logit_p)),
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

    def test_bernoulli_wrong_arguments(self):
        m = pm.Model()

        msg = "Incompatible parametrization. Can't specify both p and logit_p"
        with m:
            with pytest.raises(ValueError, match=msg):
                Bernoulli("x", p=0.5, logit_p=0)

        msg = "Incompatible parametrization. Must specify either p or logit_p"
        with m:
            with pytest.raises(ValueError, match=msg):
                Bernoulli("x")

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

    def test_constantdist(self):
        self.check_logp(Constant, I, {"c": I}, lambda value, c: np.log(c == value))

    @pytest.mark.xfail(reason="Test has not been refactored")
    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to inf issues",
    )
    def test_zeroinflatedpoisson_distribution(self):
        self.checkd(
            ZeroInflatedPoisson,
            Nat,
            {"theta": Rplus, "psi": Unit},
        )

    def test_zeroinflatedpoisson(self):
        def logp_fn(value, psi, theta):
            if value == 0:
                return np.log((1 - psi) * sp.poisson.pmf(0, theta))
            else:
                return np.log(psi * sp.poisson.pmf(value, theta))

        def logcdf_fn(value, psi, theta):
            return np.log((1 - psi) + psi * sp.poisson.cdf(value, theta))

        self.check_logp(
            ZeroInflatedPoisson,
            Nat,
            {"psi": Unit, "theta": Rplus},
            logp_fn,
        )

        self.check_logcdf(
            ZeroInflatedPoisson,
            Nat,
            {"psi": Unit, "theta": Rplus},
            logcdf_fn,
        )

        self.check_selfconsistency_discrete_logcdf(
            ZeroInflatedPoisson,
            Nat,
            {"theta": Rplus, "psi": Unit},
        )

    @pytest.mark.xfail(reason="Test not refactored yet")
    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to inf issues",
    )
    def test_zeroinflatednegativebinomial_distribution(self):
        self.checkd(
            ZeroInflatedNegativeBinomial,
            Nat,
            {"mu": Rplusbig, "alpha": Rplusbig, "psi": Unit},
        )

    def test_zeroinflatednegativebinomial(self):
        def logp_fn(value, psi, mu, alpha):
            n, p = NegativeBinomial.get_n_p(mu=mu, alpha=alpha)
            if value == 0:
                return np.log((1 - psi) * sp.nbinom.pmf(0, n, p))
            else:
                return np.log(psi * sp.nbinom.pmf(value, n, p))

        def logcdf_fn(value, psi, mu, alpha):
            n, p = NegativeBinomial.get_n_p(mu=mu, alpha=alpha)
            return np.log((1 - psi) + psi * sp.nbinom.cdf(value, n, p))

        self.check_logp(
            ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "mu": Rplusbig, "alpha": Rplusbig},
            logp_fn,
        )

        self.check_logcdf(
            ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "mu": Rplusbig, "alpha": Rplusbig},
            logcdf_fn,
        )

        self.check_selfconsistency_discrete_logcdf(
            ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "mu": Rplusbig, "alpha": Rplusbig},
        )

    @pytest.mark.xfail(reason="Test not refactored yet")
    def test_zeroinflatedbinomial_distribution(self):
        self.checkd(
            ZeroInflatedBinomial,
            Nat,
            {"n": NatSmall, "p": Unit, "psi": Unit},
        )

    def test_zeroinflatedbinomial(self):
        def logp_fn(value, psi, n, p):
            if value == 0:
                return np.log((1 - psi) * sp.binom.pmf(0, n, p))
            else:
                return np.log(psi * sp.binom.pmf(value, n, p))

        def logcdf_fn(value, psi, n, p):
            return np.log((1 - psi) + psi * sp.binom.cdf(value, n, p))

        self.check_logp(
            ZeroInflatedBinomial,
            Nat,
            {"psi": Unit, "n": NatSmall, "p": Unit},
            logp_fn,
        )

        self.check_logcdf(
            ZeroInflatedBinomial,
            Nat,
            {"psi": Unit, "n": NatSmall, "p": Unit},
            logcdf_fn,
        )

        self.check_selfconsistency_discrete_logcdf(
            ZeroInflatedBinomial,
            Nat,
            {"n": NatSmall, "p": Unit, "psi": Unit},
        )

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_mvnormal(self, n):
        self.check_logp(
            MvNormal,
            RealMatrix(5, n),
            {"mu": Vector(R, n), "tau": PdMatrix(n)},
            normal_logpdf_tau,
            extra_args={"size": 5},
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
            extra_args={"size": 5},
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
            extra_args={"size": 5},
        )
        self.check_logp(
            MvNormal,
            Vector(R, n),
            {"mu": Vector(R, n), "chol": PdMatrixChol(n)},
            normal_logpdf_chol,
            decimal=select_by_precision(float64=6, float32=0),
        )
        self.check_logp(
            MvNormal,
            Vector(R, n),
            {"mu": Vector(R, n), "chol": PdMatrixCholUpper(n)},
            normal_logpdf_chol_upper,
            decimal=select_by_precision(float64=6, float32=0),
            extra_args={"lower": False},
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to inf issues",
    )
    def test_mvnormal_indef(self):
        cov_val = np.array([[1, 0.5], [0.5, -2]])
        cov = at.matrix("cov")
        cov.tag.test_value = np.eye(2)
        mu = floatX(np.zeros(2))
        x = at.vector("x")
        x.tag.test_value = np.zeros(2)
        mvn_logp = logp(MvNormal.dist(mu=mu, cov=cov), x)
        f_logp = aesara.function([cov, x], mvn_logp)
        assert f_logp(cov_val, np.ones(2)) == -np.inf
        dlogp = at.grad(mvn_logp, cov)
        f_dlogp = aesara.function([cov, x], dlogp)
        assert not np.all(np.isfinite(f_dlogp(cov_val, np.ones(2))))

        mvn_logp = logp(MvNormal.dist(mu=mu, tau=cov), x)
        f_logp = aesara.function([cov, x], mvn_logp)
        assert f_logp(cov_val, np.ones(2)) == -np.inf
        dlogp = at.grad(mvn_logp, cov)
        f_dlogp = aesara.function([cov, x], dlogp)
        assert not np.all(np.isfinite(f_dlogp(cov_val, np.ones(2))))

    def test_mvnormal_init_fail(self):
        with Model():
            with pytest.raises(ValueError):
                x = MvNormal("x", mu=np.zeros(3), size=3)
            with pytest.raises(ValueError):
                x = MvNormal("x", mu=np.zeros(3), cov=np.eye(3), tau=np.eye(3), size=3)

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
            decimal=select_by_precision(float64=5, float32=3),
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
            decimal=select_by_precision(float64=5, float32=3),
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
            decimal=select_by_precision(float64=5, float32=3),
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
            decimal=select_by_precision(float64=5, float32=3),
        )

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("m", [3])
    @pytest.mark.parametrize("sigma", [None, 1])
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
        cov_args["size"] = 2
        chol_args["size"] = 2
        evd_args["size"] = 2

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
            extra_args={"size": 2},
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.xfail(reason="Distribution not refactored yet")
    def test_AR1(self, n):
        self.check_logp(AR1, Vector(R, n), {"k": Unit, "tau_e": Rplus}, AR1_logpdf)

    @pytest.mark.parametrize("n", [2, 3])
    def test_wishart(self, n):
        self.check_logp(
            Wishart,
            PdMatrix(n),
            {"nu": Domain([3, 4, 2000]), "V": PdMatrix(n)},
            lambda value, nu, V: scipy.stats.wishart.logpdf(value, np.int(nu), V),
        )

    @pytest.mark.parametrize("x,eta,n,lp", LKJ_CASES)
    @pytest.mark.xfail(reason="Distribution not refactored yet")
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
            d = pm.Dirichlet("d", a=a)

        # Generate sample points to test
        d_value = d.tag.value_var
        d_point = d.eval().astype("float64")
        d_point /= d_point.sum(axis=-1)[..., None]

        if hasattr(d_value.tag, "transform"):
            d_point_trans = d_value.tag.transform.forward(d, at.as_tensor(d_point)).eval()
        else:
            d_point_trans = d_point

        pymc3_res = logp(d, d_point_trans, jacobian=False).eval()
        scipy_res = np.empty_like(pymc3_res)
        for idx in np.ndindex(a.shape[:-1]):
            scipy_res[idx] = scipy.stats.dirichlet(a[idx]).logpdf(d_point[idx])

        assert_almost_equal(pymc3_res, scipy_res)

    def test_dirichlet_shape(self):
        a = at.as_tensor_variable(np.r_[1, 2])
        dir_rv = Dirichlet.dist(a)
        assert dir_rv.shape.eval() == (2,)

        with pytest.warns(DeprecationWarning), aesara.change_flags(compute_test_value="ignore"):
            dir_rv = Dirichlet.dist(at.vector())

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

    @pytest.mark.skip(reason="Moment calculations have not been refactored yet")
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
        "p, size, n",
        [
            [[0.25, 0.25, 0.25, 0.25], (4,), 2],
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
    def test_multinomial_random(self, p, size, n):
        p = np.asarray(p)
        with Model() as model:
            m = Multinomial("m", n=n, p=p, size=size)

        assert m.eval().shape == size + p.shape

    @pytest.mark.skip(reason="Moment calculations have not been refactored yet")
    def test_multinomial_mode_with_shape(self):
        n = [1, 10]
        p = np.asarray([[0.25, 0.25, 0.25, 0.25], [0.26, 0.26, 0.26, 0.22]])
        with Model() as model:
            m = Multinomial("m", n=n, p=p, size=(2, 4))
        assert_allclose(m.distribution.mode.eval().sum(axis=-1), n)

    def test_multinomial_vec(self):
        vals = np.array([[2, 4, 4], [3, 3, 4]])
        p = np.array([0.2, 0.3, 0.5])
        n = 10

        with Model() as model_single:
            Multinomial("m", n=n, p=p)

        with Model() as model_many:
            Multinomial("m", n=n, p=p, size=2)

        assert_almost_equal(
            scipy.stats.multinomial.logpmf(vals, n, p),
            np.asarray([model_single.fastlogp({"m": val}) for val in vals]),
            decimal=4,
        )

        assert_almost_equal(
            scipy.stats.multinomial.logpmf(vals, n, p),
            logp(model_many.m, vals).eval().squeeze(),
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
            Multinomial("m", n=ns, p=p)

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
            Multinomial("m", n=ns, p=ps)

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
            Multinomial("m", n=n, p=ps)

        assert_almost_equal(
            sum(multinomial_logpdf(val, n, p) for val, p in zip(vals, ps)),
            model.fastlogp({"m": vals}),
            decimal=4,
        )

    def test_batch_multinomial(self):
        n = 10
        vals = intX(np.zeros((4, 5, 3)))
        p = floatX(np.zeros_like(vals))
        inds = np.random.randint(vals.shape[-1], size=vals.shape[:-1])[..., None]
        np.put_along_axis(vals, inds, n, axis=-1)
        np.put_along_axis(p, inds, 1, axis=-1)

        dist = Multinomial.dist(n=n, p=p)
        logp_mn = at.exp(pm.logp(dist, vals)).eval()
        assert_almost_equal(
            logp_mn,
            np.ones(vals.shape[:-1]),
            decimal=select_by_precision(float64=6, float32=3),
        )

        dist = Multinomial.dist(n=n, p=p, size=2)
        sample = dist.eval()
        assert_allclose(sample, np.stack([vals, vals], axis=0))

    def test_multinomial_zero_probs(self):
        # test multinomial accepts 0 probabilities / observations:
        value = aesara.shared(np.array([0, 0, 100], dtype=int))
        logp = pm.Multinomial.logp(value=value, n=100, p=at.constant([0.0, 0.0, 1.0]))
        logp_fn = aesara.function(inputs=[], outputs=logp)
        assert logp_fn() >= 0

        value.set_value(np.array([50, 50, 0], dtype=int))
        assert np.isneginf(logp_fn())

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
        ns_dm = np.vstack((ns, n - ns)).T  # convert ns=1 to ns_dm=[1, 4], for all ns...

        bb = pm.BetaBinomial.dist(n=n, alpha=a, beta=b, size=2)
        bb_logp = logp(bb, ns).eval()

        dm = pm.DirichletMultinomial.dist(n=n, a=[a, b], size=2)
        dm_logp = logp(dm, ns_dm).eval().ravel()

        assert_almost_equal(
            dm_logp,
            bb_logp,
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_dirichlet_multinomial_vec(self):
        vals = np.array([[2, 4, 4], [3, 3, 4]])
        a = np.array([0.2, 0.3, 0.5])
        n = 10

        with Model() as model_single:
            DirichletMultinomial("m", n=n, a=a)

        with Model() as model_many:
            DirichletMultinomial("m", n=n, a=a, size=2)

        assert_almost_equal(
            np.asarray([dirichlet_multinomial_logpmf(val, n, a) for val in vals]),
            np.asarray([model_single.fastlogp({"m": val}) for val in vals]),
            decimal=4,
        )

        assert_almost_equal(
            np.asarray([dirichlet_multinomial_logpmf(val, n, a) for val in vals]),
            logp(model_many.m, vals).eval().squeeze(),
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
            DirichletMultinomial("m", n=ns, a=a)

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
            DirichletMultinomial("m", n=ns, a=as_)

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
            DirichletMultinomial("m", n=n, a=as_)

        assert_almost_equal(
            sum(dirichlet_multinomial_logpmf(val, n, a) for val, a in zip(vals, as_)),
            model.fastlogp({"m": vals}),
            decimal=4,
        )

    def test_batch_dirichlet_multinomial(self):
        # Test that DM can handle a 3d array for `a`

        # Create an almost deterministic DM by setting a to 0.001, everywhere
        # except for one category / dimension which is given the value of 1000
        n = 5
        vals = np.zeros((4, 5, 3), dtype="int32")
        a = np.zeros_like(vals, dtype=aesara.config.floatX) + 0.001
        inds = np.random.randint(vals.shape[-1], size=vals.shape[:-1])[..., None]
        np.put_along_axis(vals, inds, n, axis=-1)
        np.put_along_axis(a, inds, 1000, axis=-1)

        dist = DirichletMultinomial.dist(n=n, a=a)

        # Logp should be approx -9.98004998e-06
        dist_logp = logp(dist, vals).eval()
        expected_logp = np.full_like(dist_logp, fill_value=-9.98004998e-06)
        assert_almost_equal(
            dist_logp,
            expected_logp,
            decimal=select_by_precision(float64=6, float32=3),
        )

        # Samples should be equal given the almost deterministic DM
        dist = DirichletMultinomial.dist(n=n, a=a, size=2)
        sample = dist.eval()
        assert_allclose(sample, np.stack([vals, vals], axis=0))

    @aesara.config.change_flags(compute_test_value="raise")
    def test_categorical_bounds(self):
        with Model():
            x = Categorical("x", p=np.array([0.2, 0.3, 0.5]))
            assert np.isinf(logp(x, -1).eval())
            assert np.isinf(logp(x, 3).eval())

    @aesara.config.change_flags(compute_test_value="raise")
    def test_categorical_valid_p(self):
        with Model():
            x = Categorical("x", p=np.array([-0.2, 0.3, 0.5]))
            assert np.isinf(logp(x, 0).eval())
            assert np.isinf(logp(x, 1).eval())
            assert np.isinf(logp(x, 2).eval())
        with Model():
            # A model where p sums to 1 but contains negative values
            x = Categorical("x", p=np.array([-0.2, 0.7, 0.5]))
            assert np.isinf(logp(x, 0).eval())
            assert np.isinf(logp(x, 1).eval())
            assert np.isinf(logp(x, 2).eval())
        with Model():
            # Hard edge case from #2082
            # Early automatic normalization of p's sum would hide the negative
            # entries if there is a single or pair number of negative values
            # and the rest are zero
            x = Categorical("x", p=np.array([-1, -1, 0, 0]))
            assert np.isinf(logp(x, 0).eval())
            assert np.isinf(logp(x, 1).eval())
            assert np.isinf(logp(x, 2).eval())
            assert np.isinf(logp(x, 3).eval())

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

    @pytest.mark.xfail(reason="DensityDist no longer supported")
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
        "value,mu,sigma,nu,logcdf_val",
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
    def test_ex_gaussian_cdf(self, value, mu, sigma, nu, logcdf_val):
        """Log probabilities calculated using the pexGAUS function from the R package gamlss.
        See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or http://www.gamlss.org/."""
        assert_almost_equal(
            logcdf(ExGaussian.dist(mu=mu, sigma=sigma, nu=nu), value).eval(),
            logcdf_val,
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str((value, mu, sigma, nu, logcdf_val)),
        )

    def test_ex_gaussian_cdf_outside_edges(self):
        self.check_logcdf(
            ExGaussian,
            R,
            {"mu": R, "sigma": Rplus, "nu": Rplus},
            None,
            skip_paramdomain_inside_edge_test=True,  # Valid values are tested above
        )

    @pytest.mark.skipif(condition=(aesara.config.floatX == "float32"), reason="Fails on float32")
    def test_vonmises(self):
        self.check_logp(
            VonMises,
            R,
            {"mu": Circ, "kappa": Rplus},
            lambda value, mu, kappa: floatX(sp.vonmises.logpdf(value, kappa, loc=mu)),
        )

    def test_gumbel(self):
        self.check_logp(
            Gumbel,
            R,
            {"mu": R, "beta": Rplusbig},
            lambda value, mu, beta: sp.gumbel_r.logpdf(value, loc=mu, scale=beta),
        )
        self.check_logcdf(
            Gumbel,
            R,
            {"mu": R, "beta": Rplusbig},
            lambda value, mu, beta: sp.gumbel_r.logcdf(value, loc=mu, scale=beta),
        )

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

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Some combinations underflow to -inf in float32 in pymc version",
    )
    def test_rice(self):
        self.check_logp(
            Rice,
            Rplus,
            {"b": Rplus, "sigma": Rplusbig},
            lambda value, b, sigma: sp.rice.logpdf(value, b=b, loc=0, scale=sigma),
        )
        if aesara.config.floatX == "float32":
            raise Exception("Flaky test: It passed this time, but XPASS is not allowed.")

    def test_rice_nu(self):
        self.check_logp(
            Rice,
            Rplus,
            {"nu": Rplus, "sigma": Rplusbig},
            lambda value, nu, sigma: sp.rice.logpdf(value, b=nu / sigma, loc=0, scale=sigma),
        )

    def test_moyal_logp(self):
        # Using a custom domain, because the standard `R` domain undeflows with scipy in float64
        value_domain = Domain([-inf, -1.5, -1, -0.01, 0.0, 0.01, 1, 1.5, inf])
        self.check_logp(
            Moyal,
            value_domain,
            {"mu": R, "sigma": Rplusbig},
            lambda value, mu, sigma: floatX(sp.moyal.logpdf(value, mu, sigma)),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Pymc3 underflows earlier than scipy on float32",
    )
    def test_moyal_logcdf(self):
        self.check_logcdf(
            Moyal,
            R,
            {"mu": R, "sigma": Rplusbig},
            lambda value, mu, sigma: floatX(sp.moyal.logcdf(value, mu, sigma)),
        )
        if aesara.config.floatX == "float32":
            raise Exception("Flaky test: It passed this time, but XPASS is not allowed.")

    @pytest.mark.skipif(condition=(aesara.config.floatX == "float32"), reason="Fails on float32")
    def test_interpolated(self):
        for mu in R.vals:
            for sigma in Rplus.vals:
                # pylint: disable=cell-var-from-loop
                xmin = mu - 5 * sigma
                xmax = mu + 5 * sigma

                from pymc3.distributions.continuous import interpolated

                class TestedInterpolated(Interpolated):
                    rv_op = interpolated

                    @classmethod
                    def dist(cls, **kwargs):
                        x_points = np.linspace(xmin, xmax, 100000)
                        pdf_points = sp.norm.pdf(x_points, loc=mu, scale=sigma)
                        return super().dist(x_points=x_points, pdf_points=pdf_points, **kwargs)

                def ref_pdf(value):
                    return np.where(
                        np.logical_and(value >= xmin, value <= xmax),
                        sp.norm.logpdf(value, mu, sigma),
                        -np.inf * np.ones(value.shape),
                    )

                self.check_logp(TestedInterpolated, R, {}, ref_pdf)


class TestBound:
    """Tests for pm.Bound distribution"""

    def test_continuous(self):
        with Model() as model:
            dist = Normal.dist(mu=0, sigma=1)
            UnboundedNormal = Bound("unbound", dist, transform=None)
            InfBoundedNormal = Bound("infbound", dist, lower=-np.inf, upper=np.inf, transform=None)
            LowerNormal = Bound("lower", dist, lower=0, transform=None)
            UpperNormal = Bound("upper", dist, upper=0, transform=None)
            BoundedNormal = Bound("bounded", dist, lower=1, upper=10, transform=None)
            LowerNormalTransform = Bound("lowertrans", dist, lower=1)
            UpperNormalTransform = Bound("uppertrans", dist, upper=10)
            BoundedNormalTransform = Bound("boundedtrans", dist, lower=1, upper=10)

        assert logpt(LowerNormal, -1).eval() == -np.inf
        assert logpt(UpperNormal, 1).eval() == -np.inf
        assert logpt(BoundedNormal, 0).eval() == -np.inf
        assert logpt(BoundedNormal, 11).eval() == -np.inf

        assert logpt(UnboundedNormal, 0).eval() != -np.inf
        assert logpt(UnboundedNormal, 11).eval() != -np.inf
        assert logpt(InfBoundedNormal, 0).eval() != -np.inf
        assert logpt(InfBoundedNormal, 11).eval() != -np.inf

        assert logpt(LowerNormalTransform, -1).eval() != -np.inf
        assert logpt(UpperNormalTransform, 1).eval() != -np.inf
        assert logpt(BoundedNormalTransform, 0).eval() != -np.inf
        assert logpt(BoundedNormalTransform, 11).eval() != -np.inf

        assert np.allclose(
            logpt(UnboundedNormal, 5).eval(), Normal.logp(value=5, mu=0, sigma=1).eval()
        )
        assert np.allclose(logpt(LowerNormal, 5).eval(), Normal.logp(value=5, mu=0, sigma=1).eval())
        assert np.allclose(
            logpt(UpperNormal, -5).eval(), Normal.logp(value=-5, mu=0, sigma=1).eval()
        )
        assert np.allclose(
            logpt(BoundedNormal, 5).eval(), Normal.logp(value=5, mu=0, sigma=1).eval()
        )

    def test_discrete(self):
        with Model() as model:
            dist = Poisson.dist(mu=4)
            UnboundedPoisson = Bound("unbound", dist)
            LowerPoisson = Bound("lower", dist, lower=1)
            UpperPoisson = Bound("upper", dist, upper=10)
            BoundedPoisson = Bound("bounded", dist, lower=1, upper=10)

        assert logpt(LowerPoisson, 0).eval() == -np.inf
        assert logpt(UpperPoisson, 11).eval() == -np.inf
        assert logpt(BoundedPoisson, 0).eval() == -np.inf
        assert logpt(BoundedPoisson, 11).eval() == -np.inf

        assert logpt(UnboundedPoisson, 0).eval() != -np.inf
        assert logpt(UnboundedPoisson, 11).eval() != -np.inf

        assert np.allclose(logpt(UnboundedPoisson, 5).eval(), Poisson.logp(value=5, mu=4).eval())
        assert np.allclose(logpt(UnboundedPoisson, 5).eval(), Poisson.logp(value=5, mu=4).eval())
        assert np.allclose(logpt(UnboundedPoisson, 5).eval(), Poisson.logp(value=5, mu=4).eval())
        assert np.allclose(logpt(UnboundedPoisson, 5).eval(), Poisson.logp(value=5, mu=4).eval())

    def create_invalid_distribution(self):
        class MyNormal(RandomVariable):
            name = "my_normal"
            ndim_supp = 0
            ndims_params = [0, 0]
            dtype = "floatX"

        my_normal = MyNormal()

        class InvalidDistribution(Distribution):
            rv_op = my_normal

            @classmethod
            def dist(cls, mu=0, sigma=1, **kwargs):
                return super().dist([mu, sigma], **kwargs)

        return InvalidDistribution

    def test_arguments_checks(self):
        msg = "Observed Bound distributions are not supported"
        with pm.Model() as m:
            x = pm.Normal("x", 0, 1)
            with pytest.raises(ValueError, match=msg):
                pm.Bound("bound", x, observed=5)

        msg = "Cannot transform discrete variable."
        with pm.Model() as m:
            x = pm.Poisson.dist(0.5)
            with pytest.raises(ValueError, match=msg):
                pm.Bound("bound", x, transform=pm.transforms.interval)

        msg = "Given dims do not exist in model coordinates."
        with pm.Model() as m:
            x = pm.Poisson.dist(0.5)
            with pytest.raises(ValueError, match=msg):
                pm.Bound("bound", x, dims="random_dims")

        msg = "The distribution passed into `Bound` was already registered"
        with pm.Model() as m:
            x = pm.Normal("x", 0, 1)
            with pytest.raises(ValueError, match=msg):
                pm.Bound("bound", x)

        msg = "Passing a distribution class to `Bound` is no longer supported"
        with pm.Model() as m:
            with pytest.raises(ValueError, match=msg):
                pm.Bound("bound", pm.Normal)

        msg = "Bounding of MultiVariate RVs is not yet supported"
        with pm.Model() as m:
            x = pm.MvNormal.dist(np.zeros(3), np.eye(3))
            with pytest.raises(NotImplementedError, match=msg):
                pm.Bound("bound", x)

        msg = "must be a Discrete or Continuous distribution subclass"
        with pm.Model() as m:
            x = self.create_invalid_distribution().dist()
            with pytest.raises(ValueError, match=msg):
                pm.Bound("bound", x)

    def test_invalid_sampling(self):
        msg = "Cannot sample from a bounded variable"
        with pm.Model() as m:
            dist = Normal.dist(mu=0, sigma=1)
            BoundedNormal = Bound("bounded", dist, lower=1, upper=10)
            with pytest.raises(NotImplementedError, match=msg):
                pm.sample_prior_predictive()

    def test_bound_shapes(self):
        with pm.Model(coords={"sample": np.ones((2, 5))}) as m:
            dist = Normal.dist(mu=0, sigma=1)
            bound_sized = Bound("boundedsized", dist, lower=1, upper=10, size=(4, 5))
            bound_shaped = Bound("boundedshaped", dist, lower=1, upper=10, shape=(3, 5))
            bound_dims = Bound("boundeddims", dist, lower=1, upper=10, dims="sample")

        dist_size = m.initial_point["boundedsized_interval__"].shape
        dist_shape = m.initial_point["boundedshaped_interval__"].shape
        dist_dims = m.initial_point["boundeddims_interval__"].shape

        assert dist_size == (4, 5)
        assert dist_shape == (3, 5)
        assert dist_dims == (2, 5)

    def test_bound_dist(self):
        # Continuous
        bound = pm.Bound.dist(pm.Normal.dist(0, 1), lower=0)
        assert pm.logp(bound, -1).eval() == -np.inf
        assert np.isclose(pm.logp(bound, 1).eval(), scipy.stats.norm(0, 1).logpdf(1))

        # Discrete
        bound = pm.Bound.dist(pm.Poisson.dist(1), lower=2)
        assert pm.logp(bound, 1).eval() == -np.inf
        assert np.isclose(pm.logp(bound, 2).eval(), scipy.stats.poisson(1).logpmf(2))

    def test_array_bound(self):
        with Model() as model:
            dist = Normal.dist()
            LowerPoisson = Bound("lower", dist, lower=[1, None], transform=None)
            UpperPoisson = Bound("upper", dist, upper=[np.inf, 10], transform=None)
            BoundedPoisson = Bound("bounded", dist, lower=[1, 2], upper=[9, 10], transform=None)

        first, second = logpt(LowerPoisson, [0, 0]).eval()
        assert first == -np.inf
        assert second != -np.inf

        first, second = logpt(UpperPoisson, [11, 11]).eval()
        assert first != -np.inf
        assert second == -np.inf

        first, second = logpt(BoundedPoisson, [1, 1]).eval()
        assert first != -np.inf
        assert second == -np.inf

        first, second = logpt(BoundedPoisson, [10, 10]).eval()
        assert first == -np.inf
        assert second != -np.inf


class TestBoundedContinuous:
    def get_dist_params_and_interval_bounds(self, model, rv_name):
        interval_rv = model.named_vars[f"{rv_name}_interval__"]
        rv = model.named_vars[rv_name]
        dist_params = rv.owner.inputs[3:]
        lower_interval, upper_interval = interval_rv.tag.transform.param_extract_fn(rv)
        return (
            dist_params,
            lower_interval,
            upper_interval,
        )

    def test_upper_bounded(self):
        bounded_rv_name = "lower_bounded"
        with Model() as model:
            TruncatedNormal(bounded_rv_name, mu=1, sigma=2, lower=None, upper=3)
        (
            (_, _, lower, upper),
            lower_interval,
            upper_interval,
        ) = self.get_dist_params_and_interval_bounds(model, bounded_rv_name)
        assert lower.value == -np.inf
        assert upper.value == 3
        assert lower_interval is None
        assert upper_interval.value == 3

    def test_lower_bounded(self):
        bounded_rv_name = "upper_bounded"
        with Model() as model:
            TruncatedNormal(bounded_rv_name, mu=1, sigma=2, lower=-2, upper=None)
        (
            (_, _, lower, upper),
            lower_interval,
            upper_interval,
        ) = self.get_dist_params_and_interval_bounds(model, bounded_rv_name)
        assert lower.value == -2
        assert upper.value == np.inf
        assert lower_interval.value == -2
        assert upper_interval is None

    def test_lower_bounded_vector(self):
        bounded_rv_name = "upper_bounded"
        with Model() as model:
            TruncatedNormal(
                bounded_rv_name,
                mu=np.array([1, 1]),
                sigma=np.array([2, 3]),
                lower=np.array([-1.0, 0]),
                upper=None,
            )
        (
            (_, _, lower, upper),
            lower_interval,
            upper_interval,
        ) = self.get_dist_params_and_interval_bounds(model, bounded_rv_name)

        assert np.array_equal(lower.value, [-1, 0])
        assert upper.value == np.inf
        assert np.array_equal(lower_interval.value, [-1, 0])
        assert upper_interval is None

    def test_lower_bounded_broadcasted(self):
        bounded_rv_name = "upper_bounded"
        with Model() as model:
            TruncatedNormal(
                bounded_rv_name,
                mu=np.array([1, 1]),
                sigma=np.array([2, 3]),
                lower=-1,
                upper=np.array([np.inf, np.inf]),
            )
        (
            (_, _, lower, upper),
            lower_interval,
            upper_interval,
        ) = self.get_dist_params_and_interval_bounds(model, bounded_rv_name)

        assert lower.value == -1
        assert np.array_equal(upper.value, [np.inf, np.inf])
        assert lower_interval.value == -1
        assert upper_interval is None


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
            # TODO: some variables commented out here as they're not working properly
            # in v4 yet (9-jul-2021), so doesn't make sense to test str/latex for them

            # Priors for unknown model parameters
            alpha = Normal("alpha", mu=0, sigma=10)
            b = Normal("beta", mu=0, sigma=10, size=(2,), observed=beta)
            sigma = HalfNormal("sigma", sigma=1)

            # Test Cholesky parameterization
            Z = MvNormal("Z", mu=np.zeros(2), chol=np.eye(2), size=(2,))

            # NegativeBinomial representations to test issue 4186
            # nb1 = pm.NegativeBinomial(
            #     "nb_with_mu_alpha", mu=pm.Normal("nbmu"), alpha=pm.Gamma("nbalpha", mu=6, sigma=1)
            # )
            nb2 = pm.NegativeBinomial("nb_with_p_n", p=pm.Uniform("nbp"), n=10)

            # Expected value of outcome
            mu = Deterministic("mu", floatX(alpha + at.dot(X, b)))

            # add a bounded variable as well
            # bound_var = Bound(Normal, lower=1.0)("bound_var", mu=0, sigma=10)

            # KroneckerNormal
            n, m = 3, 4
            covs = [np.eye(n), np.eye(m)]
            kron_normal = KroneckerNormal("kron_normal", mu=np.zeros(n * m), covs=covs, size=n * m)

            # MatrixNormal
            # matrix_normal = MatrixNormal(
            #     "mat_normal",
            #     mu=np.random.normal(size=n),
            #     rowcov=np.eye(n),
            #     colchol=np.linalg.cholesky(np.eye(n)),
            #     size=(n, n),
            # )

            # DirichletMultinomial
            dm = DirichletMultinomial("dm", n=5, a=[1, 1, 1], size=(2, 3))

            # Likelihood (sampling distribution) of observations
            Y_obs = Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

            # add a potential as well
            pot = Potential("pot", mu ** 2)

        self.distributions = [alpha, sigma, mu, b, Z, nb2, Y_obs, pot]
        self.deterministics_or_potentials = [mu, pot]
        # tuples of (formatting, include_params
        self.formats = [("plain", True), ("plain", False), ("latex", True), ("latex", False)]
        self.expected = {
            ("plain", True): [
                r"alpha ~ N(0, 10)",
                r"sigma ~ N**+(0, 1)",
                r"mu ~ Deterministic(f(beta, alpha))",
                r"beta ~ N(0, 10)",
                r"Z ~ N(<constant>, f())",
                r"nb_with_p_n ~ NB(10, nbp)",
                r"Y_obs ~ N(mu, sigma)",
                r"pot ~ Potential(f(beta, alpha))",
            ],
            ("plain", False): [
                r"alpha ~ N",
                r"sigma ~ N**+",
                r"mu ~ Deterministic",
                r"beta ~ N",
                r"Z ~ N",
                r"nb_with_p_n ~ NB",
                r"Y_obs ~ N",
                r"pot ~ Potential",
            ],
            ("latex", True): [
                r"$\text{alpha} \sim \operatorname{N}(0,~10)$",
                r"$\text{sigma} \sim \operatorname{N^{+}}(0,~1)$",
                r"$\text{mu} \sim \operatorname{Deterministic}(f(\text{beta},~\text{alpha}))$",
                r"$\text{beta} \sim \operatorname{N}(0,~10)$",
                r"$\text{Z} \sim \operatorname{N}(\text{<constant>},~f())$",
                r"$\text{nb_with_p_n} \sim \operatorname{NB}(10,~\text{nbp})$",
                r"$\text{Y_obs} \sim \operatorname{N}(\text{mu},~\text{sigma})$",
                r"$\text{pot} \sim \operatorname{Potential}(f(\text{beta},~\text{alpha}))$",
            ],
            ("latex", False): [
                r"$\text{alpha} \sim \operatorname{N}$",
                r"$\text{sigma} \sim \operatorname{N^{+}}$",
                r"$\text{mu} \sim \operatorname{Deterministic}$",
                r"$\text{beta} \sim \operatorname{N}$",
                r"$\text{Z} \sim \operatorname{N}$",
                r"$\text{nb_with_p_n} \sim \operatorname{NB}$",
                r"$\text{Y_obs} \sim \operatorname{N}$",
                r"$\text{pot} \sim \operatorname{Potential}$",
            ],
        }

    def test__repr_latex_(self):
        for distribution, tex in zip(self.distributions, self.expected[("latex", True)]):
            assert distribution._repr_latex_() == tex

        model_tex = self.model._repr_latex_()

        # make sure each variable is in the model
        for tex in self.expected[("latex", True)]:
            for segment in tex.strip("$").split(r"\sim"):
                assert segment in model_tex

    def test_str_repr(self):
        for str_format in self.formats:
            for dist, text in zip(self.distributions, self.expected[str_format]):
                assert dist.str_repr(*str_format) == text

            model_text = self.model.str_repr(*str_format)
            for text in self.expected[str_format]:
                if str_format[0] == "latex":
                    for segment in text.strip("$").split(r"\sim"):
                        assert segment in model_text
                else:
                    assert text in model_text


def test_discrete_trafo():
    with Model():
        with pytest.raises(ValueError) as err:
            Binomial("a", n=5, p=0.5, transform="log")
        err.match("Transformations for discrete distributions")


# TODO: Is this test working as expected / still relevant?
@pytest.mark.parametrize("shape", [tuple(), (1,), (3, 1), (3, 2)], ids=str)
def test_orderedlogistic_dimensions(shape):
    # Test for issue #3535
    loge = np.log10(np.exp(1))
    size = 7
    p = np.ones(shape + (10,)) / 10
    cutpoints = np.tile(logit(np.linspace(0, 1, 11)[1:-1]), shape + (1,))
    obs = np.random.randint(0, 2, size=(size,) + shape)
    with Model():
        ol = OrderedLogistic(
            "ol",
            eta=np.zeros(shape),
            cutpoints=cutpoints,
            observed=obs,
        )
        c = Categorical(
            "c",
            p=p,
            observed=obs,
        )
    ologp = logpt_sum(ol, np.ones_like(obs)).eval() * loge
    clogp = logpt_sum(c, np.ones_like(obs)).eval() * loge
    expected = -np.prod((size,) + shape)

    assert c.owner.inputs[3].ndim == (len(shape) + 1)
    assert np.allclose(clogp, expected)
    assert ol.owner.inputs[3].ndim == (len(shape) + 1)
    assert np.allclose(ologp, expected)


def test_ordered_multinomial_probs():
    with pm.Model() as m:
        pm.OrderedMultinomial("om_p", n=1000, cutpoints=np.array([-2, 0, 2]), eta=0)
        pm.OrderedMultinomial(
            "om_no_p", n=1000, cutpoints=np.array([-2, 0, 2]), eta=0, compute_p=False
        )
    assert len(m.deterministics) == 1

    x = pm.OrderedMultinomial.dist(n=1000, cutpoints=np.array([-2, 0, 2]), eta=0)
    assert isinstance(x, TensorVariable)


def test_ordered_logistic_probs():
    with pm.Model() as m:
        pm.OrderedLogistic("ol_p", cutpoints=np.array([-2, 0, 2]), eta=0)
        pm.OrderedLogistic("ol_no_p", cutpoints=np.array([-2, 0, 2]), eta=0, compute_p=False)
    assert len(m.deterministics) == 1

    x = pm.OrderedLogistic.dist(cutpoints=np.array([-2, 0, 2]), eta=0)
    assert isinstance(x, TensorVariable)


def test_ordered_probit_probs():
    with pm.Model() as m:
        pm.OrderedProbit("op_p", cutpoints=np.array([-2, 0, 2]), eta=0)
        pm.OrderedProbit("op_no_p", cutpoints=np.array([-2, 0, 2]), eta=0, compute_p=False)
    assert len(m.deterministics) == 1

    x = pm.OrderedProbit.dist(cutpoints=np.array([-2, 0, 2]), eta=0)
    assert isinstance(x, TensorVariable)


@pytest.mark.parametrize(
    "sparse, size",
    [(False, ()), (False, (1,)), (False, (4,)), (False, (4, 4, 4)), (True, ()), (True, (4,))],
    ids=str,
)
def test_car_logp(sparse, size):
    """
    Tests the log probability function for the CAR distribution by checking
    against Scipy's multivariate normal logpdf, up to an additive constant.
    The formula used by the CAR logp implementation omits several additive terms.
    """
    np.random.seed(1)

    # d x d adjacency matrix for a square (d=4) of rook-adjacent sites
    W = np.array(
        [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
    )

    tau = 2
    alpha = 0.5
    mu = np.zeros(4)

    xs = np.random.randn(*(size + mu.shape))

    # Compute CAR covariance matrix and resulting MVN logp
    D = W.sum(axis=0)
    prec = tau * (np.diag(D) - alpha * W)
    cov = np.linalg.inv(prec)
    scipy_logp = scipy.stats.multivariate_normal.logpdf(xs, mu, cov)

    W = aesara.tensor.as_tensor_variable(W)
    if sparse:
        W = aesara.sparse.csr_from_dense(W)

    car_dist = CAR.dist(mu, W, alpha, tau, size=size)
    car_logp = logp(car_dist, xs).eval()

    # Check to make sure that the CAR and MVN log PDFs are equivalent
    # up to an additive constant which is independent of the CAR parameters
    delta_logp = scipy_logp - car_logp

    # Check to make sure all the delta values are identical.
    tol = 1e-08
    if aesara.config.floatX == "float32":
        tol = 1e-5
    assert np.allclose(delta_logp - delta_logp[0], 0.0, atol=tol)


@pytest.mark.parametrize(
    "sparse",
    [False, True],
    ids=str,
)
def test_car_matrix_check(sparse):
    """
    Tests the check of W matrix symmetry in CARRV.make_node.
    """
    np.random.seed(1)
    tau = 2
    alpha = 0.5
    mu = np.zeros(4)
    xs = np.random.randn(*mu.shape)

    # non-symmetric matrix
    W = np.array(
        [[0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
    )
    W = aesara.tensor.as_tensor_variable(W)
    if sparse:
        W = aesara.sparse.csr_from_dense(W)

    car_dist = CAR.dist(mu, W, alpha, tau)
    with pytest.raises(AssertionError, match="W must be a symmetric adjacency matrix"):
        logp(car_dist, xs).eval()

    # W.ndim != 2
    if not sparse:
        W = np.array([0.0, 1.0, 2.0, 0.0])
        W = aesara.tensor.as_tensor_variable(W)
        with pytest.raises(ValueError, match="W must be a matrix"):
            car_dist = CAR.dist(mu, W, alpha, tau)


class TestBugfixes:
    @pytest.mark.parametrize("dist_cls,kwargs", [(MvNormal, dict()), (MvStudentT, dict(nu=2))])
    @pytest.mark.parametrize("dims", [1, 2, 4])
    def test_issue_3051(self, dims, dist_cls, kwargs):
        mu = np.repeat(0, dims)
        d = dist_cls.dist(mu=mu, cov=np.eye(dims), **kwargs, size=(20))

        X = np.random.normal(size=(20, dims))
        actual_t = logp(d, X)
        assert isinstance(actual_t, TensorVariable)
        actual_a = actual_t.eval()
        assert isinstance(actual_a, np.ndarray)
        assert actual_a.shape == (X.shape[0],)

    def test_issue_4499(self):
        # Test for bug in Uniform and DiscreteUniform logp when setting check_bounds = False
        # https://github.com/pymc-devs/pymc3/issues/4499
        with pm.Model(check_bounds=False) as m:
            x = pm.Uniform("x", 0, 2, size=10, transform=None)
        assert_almost_equal(m.logp({"x": np.ones(10)}), -np.log(2) * 10)

        with pm.Model(check_bounds=False) as m:
            x = pm.DiscreteUniform("x", 0, 1, size=10)
        assert_almost_equal(m.logp({"x": np.ones(10)}), -np.log(2) * 10)

        with pm.Model(check_bounds=False) as m:
            x = pm.Constant("x", 1, size=10)
        assert_almost_equal(m.logp({"x": np.ones(10)}), 0 * 10)


@pytest.mark.xfail(reason="DensityDist no longer supported")
def test_serialize_density_dist():
    def func(x):
        return -2 * (x ** 2).sum()

    with pm.Model():
        pm.Normal("x")
        y = pm.DensityDist("y", func)
        pm.sample(draws=5, tune=1, mp_ctx="spawn")

    import cloudpickle

    cloudpickle.loads(cloudpickle.dumps(y))


def test_distinct_rvs():
    """Make sure `RandomVariable`s generated using a `Model`'s default RNG state all have distinct states."""

    with pm.Model(rng_seeder=np.random.RandomState(2023532)) as model:
        X_rv = pm.Normal("x")
        Y_rv = pm.Normal("y")

        pp_samples = pm.sample_prior_predictive(samples=2)

    assert X_rv.owner.inputs[0] != Y_rv.owner.inputs[0]

    assert len(model.rng_seq) == 2

    with pm.Model(rng_seeder=np.random.RandomState(2023532)):
        X_rv = pm.Normal("x")
        Y_rv = pm.Normal("y")

        pp_samples_2 = pm.sample_prior_predictive(samples=2)

    assert np.array_equal(pp_samples["y"], pp_samples_2["y"])
