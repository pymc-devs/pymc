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

from aeppl.logprob import ParameterValueError
from aesara.tensor.random.utils import broadcast_params

from pymc.distributions.continuous import get_tau_sigma
from pymc.util import UNSET

try:
    from polyagamma import polyagamma_cdf, polyagamma_pdf

    _polyagamma_not_installed = False
except ImportError:  # pragma: no cover

    _polyagamma_not_installed = True

    def polyagamma_pdf(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")

    def polyagamma_cdf(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")


from contextlib import ExitStack as does_not_raise

import pytest
import scipy.stats
import scipy.stats.distributions as sp

from aesara.compile.mode import Mode
from aesara.graph.basic import ancestors
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable
from numpy import array, inf
from numpy.testing import assert_almost_equal, assert_equal
from scipy import integrate
from scipy.special import erf, gammaln, logit

import pymc as pm

from pymc.aesaraf import floatX, intX
from pymc.distributions import (
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
    StickBreakingWeights,
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
    joint_logpt,
    logcdf,
    logp,
)
from pymc.distributions.shape_utils import to_tuple
from pymc.math import kronecker
from pymc.model import Deterministic, Model, Point, Potential
from pymc.tests.helpers import select_by_precision
from pymc.vartypes import continuous_types, discrete_types


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
    def __init__(self, vals, dtype=aesara.config.floatX, edges=None, shape=None):
        # Infinity values must be kept as floats
        vals = [array(v, dtype=dtype) if np.all(np.isfinite(v)) else floatX(v) for v in vals]

        if edges is None:
            edges = array(vals[0]), array(vals[-1])
            vals = vals[1:-1]
        else:
            edges = list(edges)
            if edges[0] is None:
                edges[0] = np.full_like(vals[0], -np.inf)
            if edges[1] is None:
                edges[1] = np.full_like(vals[0], np.inf)
            edges = tuple(edges)

        if not vals:
            raise ValueError(
                f"Domain has no values left after removing edges: {edges}.\n"
                "You can duplicate the edge values or explicitly specify the edges with the edge keyword.\n"
                f"For example: `Domain([{edges[0]}, {edges[0]}, {edges[1]}, {edges[1]}])`"
            )

        if shape is None:
            shape = vals[0].shape

        self.vals = vals
        self.shape = shape
        self.lower, self.upper = edges
        self.dtype = dtype

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


@pytest.mark.parametrize(
    "values, edges, expectation",
    [
        ([], None, pytest.raises(IndexError)),
        ([], (0, 0), pytest.raises(ValueError)),
        ([0], None, pytest.raises(ValueError)),
        ([0], (0, 0), does_not_raise()),
        ([-1, 1], None, pytest.raises(ValueError)),
        ([-1, 0, 1], None, does_not_raise()),
    ],
)
def test_domain(values, edges, expectation):
    with expectation:
        Domain(values, edges=edges)


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

Runif = Domain([-np.inf, -0.4, 0, 0.4, np.inf])
Rdunif = Domain([-np.inf, -1, 0, 1, np.inf], "int64")
Rplusunif = Domain([0, 0.5, inf])
Rplusdunif = Domain([0, 10, np.inf], "int64")

I = Domain([-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf], "int64")

NatSmall = Domain([0, 3, 4, 5, np.inf], "int64")
Nat = Domain([0, 1, 2, 3, np.inf], "int64")
NatBig = Domain([0, 1, 2, 3, 5000, np.inf], "int64")
PosNat = Domain([1, 2, 3, np.inf], "int64")

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


def _dirichlet_multinomial_logpmf(value, n, a):
    if value.sum() == n and (0 <= value).all() and (value <= n).all():
        sum_a = a.sum()
        const = gammaln(n + 1) + gammaln(sum_a) - gammaln(n + sum_a)
        series = gammaln(value + a) - gammaln(value + 1) - gammaln(a)
        return const + series.sum()
    else:
        return -inf


dirichlet_multinomial_logpmf = np.vectorize(
    _dirichlet_multinomial_logpmf, signature="(n),(),(n)->()"
)


def beta_mu_sigma(value, mu, sigma):
    kappa = mu * (1 - mu) / sigma**2 - 1
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
        cov += sigma**2 * np.eye(*cov.shape)
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


def _dirichlet_logpdf(value, a):
    # scipy.stats.dirichlet.logpdf suffers from numerical precision issues
    return -betafn(a) + logpow(value, a - 1).sum()


dirichlet_logpdf = np.vectorize(_dirichlet_logpdf, signature="(n),(n)->()")


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
    tau = tau_e * (1 - k**2)
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


def Simplex(n):
    return Domain(simplex_values(n), shape=(n,), dtype=Unit.dtype, edges=(None, None))


def MultiSimplex(n_dependent, n_independent):
    vals = []
    for simplex_value in itertools.product(simplex_values(n_dependent), repeat=n_independent):
        vals.append(np.vstack(simplex_value))

    return Domain(vals, dtype=Unit.dtype, shape=(n_independent, n_dependent))


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

    logpt_ancestors = list(ancestors([m.logpt()]))
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

    logpt_ancestors = list(ancestors([model.logpt()]))
    ops = {a.owner.op for a in logpt_ancestors if a.owner}
    assert len(ops) > 0
    assert not any(isinstance(o, RandomVariable) for o in ops)


class TestMatchesScipy:
    def check_logp(
        self,
        pymc_dist,
        domain,
        paramdomains,
        scipy_logp,
        decimal=None,
        n_samples=100,
        extra_args=None,
        scipy_args=None,
        skip_paramdomain_outside_edge_test=False,
    ):
        """
        Generic test for PyMC logp methods

        Test PyMC logp and equivalent scipy logpmf/logpdf methods give similar
        results for valid values and parameters inside the supported edges.
        Edges are excluded by default, but can be artificially included by
        creating a domain with repeated values (e.g., `Domain([0, 0, .5, 1, 1]`)

        Parameters
        ----------
        pymc_dist: PyMC distribution
        domain : Domain
            Supported domain of distribution values
        paramdomains : Dictionary of Parameter : Domain pairs
            Supported domains of distribution parameters
        scipy_logp : Scipy logpmf/logpdf method
            Scipy logp method of equivalent pymc_dist distribution
        decimal : Int
            Level of precision with which pymc_dist and scipy logp are compared.
            Defaults to 6 for float64 and 3 for float32
        n_samples : Int
            Upper limit on the number of valid domain and value combinations that
            are compared between pymc and scipy methods. If n_samples is below the
            total number of combinations, a random subset is evaluated. Setting
            n_samples = -1, will return all possible combinations. Defaults to 100
        extra_args : Dictionary with extra arguments needed to build pymc model
            Dictionary is passed to helper function `build_model` from which
            the pymc distribution logp is calculated
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

        def _model_input_dict(model, param_vars, pt):
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

        model, param_vars = build_model(pymc_dist, domain, paramdomains, extra_args)
        logp_pymc = model.compile_logp(jacobian=False)

        # Test supported value and parameters domain matches scipy
        domains = paramdomains.copy()
        domains["value"] = domain
        for pt in product(domains, n_samples=n_samples):
            pt = dict(pt)
            pt_d = _model_input_dict(model, param_vars, pt)
            pt_logp = Point(pt_d, model=model)
            pt_ref = Point(pt, filter_model_vars=False, model=model)
            assert_almost_equal(
                logp_pymc(pt_logp),
                logp_reference(pt_ref),
                decimal=decimal,
                err_msg=str(pt),
            )

        valid_value = domain.vals[0]
        valid_params = {param: paramdomain.vals[0] for param, paramdomain in paramdomains.items()}
        valid_dist = pymc_dist.dist(**valid_params, **extra_args)

        # Test pymc distribution raises ParameterValueError for scalar parameters outside
        # the supported domain edges (excluding edges)
        if not skip_paramdomain_outside_edge_test:
            # Step1: collect potential invalid parameters
            invalid_params = {param: [None, None] for param in paramdomains}
            for param, paramdomain in paramdomains.items():
                if np.ndim(paramdomain.lower) != 0:
                    continue
                if np.isfinite(paramdomain.lower):
                    invalid_params[param][0] = paramdomain.lower - 1
                if np.isfinite(paramdomain.upper):
                    invalid_params[param][1] = paramdomain.upper + 1

            # Step2: test invalid parameters, one a time
            for invalid_param, invalid_edges in invalid_params.items():
                for invalid_edge in invalid_edges:
                    if invalid_edge is None:
                        continue
                    test_params = valid_params.copy()  # Shallow copy should be okay
                    test_params[invalid_param] = at.as_tensor_variable(invalid_edge)
                    # We need to remove `Assert`s introduced by checks like
                    # `assert_negative_support` and disable test values;
                    # otherwise, we won't be able to create the `RandomVariable`
                    with aesara.config.change_flags(compute_test_value="off"):
                        invalid_dist = pymc_dist.dist(**test_params, **extra_args)
                    with aesara.config.change_flags(mode=Mode("py")):
                        with pytest.raises(ParameterValueError):
                            logp(invalid_dist, valid_value).eval()
                            pytest.fail(f"test_params={test_params}, valid_value={valid_value}")

        # Test that values outside of scalar domain support evaluate to -np.inf
        if np.ndim(domain.lower) != 0:
            return
        invalid_values = [None, None]
        if np.isfinite(domain.lower):
            invalid_values[0] = domain.lower - 1
        if np.isfinite(domain.upper):
            invalid_values[1] = domain.upper + 1

        for invalid_value in invalid_values:
            if invalid_value is None:
                continue
            with aesara.config.change_flags(mode=Mode("py")):
                assert_equal(
                    logp(valid_dist, invalid_value).eval(),
                    -np.inf,
                    err_msg=str(invalid_value),
                )

    def check_logcdf(
        self,
        pymc_dist,
        domain,
        paramdomains,
        scipy_logcdf,
        decimal=None,
        n_samples=100,
        skip_paramdomain_inside_edge_test=False,
        skip_paramdomain_outside_edge_test=False,
    ):
        """
        Generic test for PyMC logcdf methods

        The following tests are performed by default:
            1. Test PyMC logcdf and equivalent scipy logcdf methods give similar
            results for valid values and parameters inside the supported edges.
            Edges are excluded by default, but can be artificially included by
            creating a domain with repeated values (e.g., `Domain([0, 0, .5, 1, 1]`)
            Can be skipped via skip_paramdomain_inside_edge_test
            2. Test PyMC logcdf method returns -inf for invalid parameter values
            outside the supported edges. Can be skipped via skip_paramdomain_outside_edge_test
            3. Test PyMC logcdf method returns -inf and 0 for values below and
            above the supported edge, respectively, when using valid parameters.
            4. Test PyMC logcdf methods works with multiple value or returns
            default informative TypeError

        Parameters
        ----------
        pymc_dist: PyMC distribution
        domain : Domain
            Supported domain of distribution values
        paramdomains : Dictionary of Parameter : Domain pairs
            Supported domains of distribution parameters
        scipy_logcdf : Scipy logcdf method
            Scipy logcdf method of equivalent pymc_dist distribution
        decimal : Int
            Level of precision with which pymc_dist and scipy_logcdf are compared.
            Defaults to 6 for float64 and 3 for float32
        n_samples : Int
            Upper limit on the number of valid domain and value combinations that
            are compared between pymc and scipy methods. If n_samples is below the
            total number of combinations, a random subset is evaluated. Setting
            n_samples = -1, will return all possible combinations. Defaults to 100
        skip_paramdomain_inside_edge_test : Bool
            Whether to run test 1., which checks that pymc and scipy distributions
            match for valid values and parameters inside the respective domain edges
        skip_paramdomain_outside_edge_test : Bool
            Whether to run test 2., which checks that pymc distribution logcdf
            returns -inf for invalid parameter values outside the supported domain edge

        Returns
        -------

        """
        # Test pymc and scipy distributions match for values and parameters
        # within the supported domain edges (excluding edges)
        if not skip_paramdomain_inside_edge_test:
            domains = paramdomains.copy()
            domains["value"] = domain

            model, param_vars = build_model(pymc_dist, domain, paramdomains)
            rv = model["value"]
            value = model.rvs_to_values[rv]
            pymc_logcdf = model.compile_fn(logcdf(rv, value))

            if decimal is None:
                decimal = select_by_precision(float64=6, float32=3)

            for pt in product(domains, n_samples=n_samples):
                params = dict(pt)
                scipy_eval = scipy_logcdf(**params)

                value = params.pop("value")
                # Update shared parameter variables in pymc_logcdf function
                for param_name, param_value in params.items():
                    param_vars[param_name].set_value(param_value)
                pymc_eval = pymc_logcdf({"value": value})

                params["value"] = value  # for displaying in err_msg
                assert_almost_equal(
                    pymc_eval,
                    scipy_eval,
                    decimal=decimal,
                    err_msg=str(params),
                )

        valid_value = domain.vals[0]
        valid_params = {param: paramdomain.vals[0] for param, paramdomain in paramdomains.items()}
        valid_dist = pymc_dist.dist(**valid_params)

        # Test pymc distribution raises ParameterValueError for parameters outside the
        # supported domain edges (excluding edges)
        if not skip_paramdomain_outside_edge_test:
            # Step1: collect potential invalid parameters
            invalid_params = {param: [None, None] for param in paramdomains}
            for param, paramdomain in paramdomains.items():
                if np.isfinite(paramdomain.lower):
                    invalid_params[param][0] = paramdomain.lower - 1
                if np.isfinite(paramdomain.upper):
                    invalid_params[param][1] = paramdomain.upper + 1
            # Step2: test invalid parameters, one a time
            for invalid_param, invalid_edges in invalid_params.items():
                for invalid_edge in invalid_edges:
                    if invalid_edge is not None:
                        test_params = valid_params.copy()  # Shallow copy should be okay
                        test_params[invalid_param] = at.as_tensor_variable(invalid_edge)
                        # We need to remove `Assert`s introduced by checks like
                        # `assert_negative_support` and disable test values;
                        # otherwise, we won't be able to create the
                        # `RandomVariable`
                        with aesara.config.change_flags(compute_test_value="off"):
                            invalid_dist = pymc_dist.dist(**test_params)
                        with aesara.config.change_flags(mode=Mode("py")):
                            with pytest.raises(ParameterValueError):
                                logcdf(invalid_dist, valid_value).eval()

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
        if np.isfinite(domain.upper):
            above_domain = domain.upper + 1
            with aesara.config.change_flags(mode=Mode("py")):
                assert_equal(
                    logcdf(valid_dist, above_domain).eval(),
                    0,
                    err_msg=str(above_domain),
                )

        # Test that method works with multiple values or raises informative TypeError
        valid_dist = pymc_dist.dist(**valid_params, size=2)
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
        rv_op = getattr(distribution, "rv_op", None)
        if rv_op:
            assert rv_op.ndim_supp == 0

        domains = paramdomains.copy()
        domains["value"] = domain
        if decimal is None:
            decimal = select_by_precision(float64=6, float32=3)

        model, param_vars = build_model(distribution, domain, paramdomains)
        rv = model["value"]
        value = model.rvs_to_values[rv]
        dist_logcdf = model.compile_fn(logcdf(rv, value))
        dist_logp = model.compile_fn(logp(rv, value))

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

    def test_uniform(self):
        self.check_logp(
            Uniform,
            Runif,
            {"lower": -Rplusunif, "upper": Rplusunif},
            lambda value, lower, upper: sp.uniform.logpdf(value, lower, upper - lower),
            skip_paramdomain_outside_edge_test=True,
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
            skip_paramdomain_outside_edge_test=True,
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

        # Custom logcdf check for invalid parameters.
        # Invalid logp checks for triangular are being done in aeppl
        invalid_dist = Triangular.dist(lower=1, upper=0, c=0.1)
        with aesara.config.change_flags(mode=Mode("py")):
            with pytest.raises(ParameterValueError):
                logcdf(invalid_dist, 2).eval()

        invalid_dist = Triangular.dist(lower=0, upper=1, c=2.0)
        with aesara.config.change_flags(mode=Mode("py")):
            with pytest.raises(ParameterValueError):
                logcdf(invalid_dist, 2).eval()

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
            skip_paramdomain_outside_edge_test=True,
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
            Domain([-10, 0, 10], "int64"),
            {"lower": -Rplusdunif, "upper": Rplusdunif},
        )
        # Custom logp / logcdf check for invalid parameters
        invalid_dist = DiscreteUniform.dist(lower=1, upper=0)
        with aesara.config.change_flags(mode=Mode("py")):
            with pytest.raises(ParameterValueError):
                logp(invalid_dist, 0.5).eval()
            with pytest.raises(ParameterValueError):
                logcdf(invalid_dist, 2).eval()

    def test_flat(self):
        self.check_logp(Flat, R, {}, lambda value: 0)
        with Model():
            x = Flat("a")
        self.check_logcdf(Flat, R, {}, lambda value: np.log(0.5))
        # Check infinite cases individually.
        assert 0.0 == logcdf(Flat.dist(), np.inf).eval()
        assert -np.inf == logcdf(Flat.dist(), -np.inf).eval()

    def test_half_flat(self):
        self.check_logp(HalfFlat, Rplus, {}, lambda value: 0)
        with Model():
            x = HalfFlat("a", size=2)
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
            skip_paramdomain_outside_edge_test=True,
        )

        self.check_logp(
            TruncatedNormal,
            R,
            {"mu": R, "sigma": Rplusbig, "upper": Rplusbig},
            functools.partial(scipy_logp, lower=-np.inf),
            decimal=select_by_precision(float64=6, float32=1),
            skip_paramdomain_outside_edge_test=True,
        )

        self.check_logp(
            TruncatedNormal,
            R,
            {"mu": R, "sigma": Rplusbig, "lower": -Rplusbig},
            functools.partial(scipy_logp, upper=np.inf),
            decimal=select_by_precision(float64=6, float32=1),
            skip_paramdomain_outside_edge_test=True,
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
        assert_almost_equal(model.compile_logp()(pt), logp, decimal=decimals, err_msg=str(pt))

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
                np.log(a) + np.log(b) + (a - 1) * np.log(value) + (b - 1) * np.log(1 - value**a)
            )

        def scipy_log_cdf(value, a, b):
            return pm.math.log1mexp_numpy(b * np.log1p(-(value**a)), negative_input=True)

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

    @pytest.mark.xfail(
        condition=(aesara.config.floatX == "float32"),
        reason="SciPy log CDF stopped working after un-pinning NumPy version.",
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
            lambda value, mu, tau: floatX(sp.lognorm.logpdf(value, tau**-0.5, 0, np.exp(mu))),
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
            lambda value, mu, tau: sp.lognorm.logcdf(value, tau**-0.5, 0, np.exp(mu)),
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
            lambda value, nu, mu, lam: sp.t.logpdf(value, nu, mu, lam**-0.5),
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
            lambda value, nu, mu, lam: sp.t.logcdf(value, nu, mu, lam**-0.5),
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
            return sp.gamma.logpdf(value, mu**2 / sigma**2, scale=1.0 / (mu / sigma**2))

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
            {"q": Unit, "beta": NatSmall},
            discrete_weibull_logpmf,
        )
        self.check_selfconsistency_discrete_logcdf(
            DiscreteWeibull,
            Nat,
            {"q": Unit, "beta": NatSmall},
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
        self.check_logcdf(Constant, I, {"c": I}, lambda value, c: np.log(value >= c))

    def test_zeroinflatedpoisson(self):
        def logp_fn(value, psi, mu):
            if value == 0:
                return np.log((1 - psi) * sp.poisson.pmf(0, mu))
            else:
                return np.log(psi * sp.poisson.pmf(value, mu))

        def logcdf_fn(value, psi, mu):
            return np.log((1 - psi) + psi * sp.poisson.cdf(value, mu))

        self.check_logp(
            ZeroInflatedPoisson,
            Nat,
            {"psi": Unit, "mu": Rplus},
            logp_fn,
        )

        self.check_logcdf(
            ZeroInflatedPoisson,
            Nat,
            {"psi": Unit, "mu": Rplus},
            logcdf_fn,
        )

        self.check_selfconsistency_discrete_logcdf(
            ZeroInflatedPoisson,
            Nat,
            {"mu": Rplus, "psi": Unit},
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

        self.check_logp(
            ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "p": Unit, "n": NatSmall},
            lambda value, psi, p, n: np.log((1 - psi) * sp.nbinom.pmf(0, n, p))
            if value == 0
            else np.log(psi * sp.nbinom.pmf(value, n, p)),
        )

        self.check_logcdf(
            ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "mu": Rplusbig, "alpha": Rplusbig},
            logcdf_fn,
        )

        self.check_logcdf(
            ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "p": Unit, "n": NatSmall},
            lambda value, psi, p, n: np.log((1 - psi) + psi * sp.nbinom.cdf(value, n, p)),
        )

        self.check_selfconsistency_discrete_logcdf(
            ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "mu": Rplusbig, "alpha": Rplusbig},
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
        with pytest.raises(ParameterValueError):
            f_logp(cov_val, np.ones(2))
        dlogp = at.grad(mvn_logp, cov)
        f_dlogp = aesara.function([cov, x], dlogp)
        assert not np.all(np.isfinite(f_dlogp(cov_val, np.ones(2))))

        mvn_logp = logp(MvNormal.dist(mu=mu, tau=cov), x)
        f_logp = aesara.function([cov, x], mvn_logp)
        with pytest.raises(ParameterValueError):
            f_logp(cov_val, np.ones(2))
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
            {"nu": Domain([0, 3, 4, np.inf], "int64"), "V": PdMatrix(n)},
            lambda value, nu, V: scipy.stats.wishart.logpdf(value, np.int(nu), V),
        )

    @pytest.mark.parametrize("x,eta,n,lp", LKJ_CASES)
    def test_lkjcorr(self, x, eta, n, lp):
        with Model() as model:
            LKJCorr("lkj", eta=eta, n=n, transform=None)

        pt = {"lkj": x}
        decimals = select_by_precision(float64=6, float32=4)
        assert_almost_equal(model.compile_logp()(pt), lp, decimal=decimals, err_msg=str(pt))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_dirichlet(self, n):
        self.check_logp(
            Dirichlet,
            Simplex(n),
            {"a": Vector(Rplus, n)},
            dirichlet_logpdf,
        )

    def test_dirichlet_invalid(self):
        # Test non-scalar invalid parameters/values
        value = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])

        invalid_dist = Dirichlet.dist(a=[-1, 1, 2], size=2)
        with pytest.raises(ParameterValueError):
            pm.logp(invalid_dist, value).eval()

        value[1] -= 1
        valid_dist = Dirichlet.dist(a=[1, 1, 1])
        assert np.all(np.isfinite(pm.logp(valid_dist, value).eval()) == np.array([True, False]))

    @pytest.mark.parametrize(
        "a",
        [
            ([2, 3, 5]),
            ([[2, 3, 5], [9, 19, 3]]),
            (np.abs(np.random.randn(2, 2, 4)) + 1),
        ],
    )
    @pytest.mark.parametrize("extra_size", [(2,), (1, 2), (2, 4, 3)])
    def test_dirichlet_vectorized(self, a, extra_size):
        a = floatX(np.array(a))
        size = extra_size + a.shape[:-1]

        dir = pm.Dirichlet.dist(a=a, size=size)
        vals = dir.eval()

        assert_almost_equal(
            dirichlet_logpdf(vals, a),
            pm.logp(dir, vals).eval(),
            decimal=4,
            err_msg=f"vals={vals}",
        )

    @pytest.mark.parametrize("n", [2, 3])
    def test_multinomial(self, n):
        self.check_logp(
            Multinomial,
            Vector(Nat, n),
            {"p": Simplex(n), "n": Nat},
            lambda value, n, p: scipy.stats.multinomial.logpmf(value, n, p),
        )

    def test_multinomial_invalid_value(self):
        # Test passing non-scalar invalid parameters/values to an otherwise valid Multinomial,
        # evaluates to -inf
        value = np.array([[1, 2, 2], [3, -1, 0]])
        valid_dist = Multinomial.dist(n=5, p=np.ones(3) / 3)
        assert np.all(np.isfinite(pm.logp(valid_dist, value).eval()) == np.array([True, False]))

    def test_multinomial_negative_p(self):
        # test passing a list/numpy with negative p raises an immediate error
        with pytest.raises(ValueError, match="[-1, 1, 1]"):
            with Model() as model:
                x = Multinomial("x", n=5, p=[-1, 1, 1])

    def test_multinomial_p_not_normalized(self):
        # test UserWarning is raised for p vals that sum to more than 1
        # and normaliation is triggered
        with pytest.warns(UserWarning, match="[5]"):
            with pm.Model() as m:
                x = pm.Multinomial("x", n=5, p=[1, 1, 1, 1, 1])
        # test stored p-vals have been normalised
        assert np.isclose(m.x.owner.inputs[4].sum().eval(), 1.0)

    def test_multinomial_negative_p_symbolic(self):
        # Passing symbolic negative p does not raise an immediate error, but evaluating
        # logp raises a ParameterValueError
        with pytest.raises(ParameterValueError):
            value = np.array([[1, 1, 1]])
            invalid_dist = pm.Multinomial.dist(n=1, p=at.as_tensor_variable([-1, 0.5, 0.5]))
            pm.logp(invalid_dist, value).eval()

    def test_multinomial_p_not_normalized_symbolic(self):
        # Passing symbolic p that do not add up to on does not raise any warning, but evaluating
        # logp raises a ParameterValueError
        with pytest.raises(ParameterValueError):
            value = np.array([[1, 1, 1]])
            invalid_dist = pm.Multinomial.dist(n=1, p=at.as_tensor_variable([1, 0.5, 0.5]))
            pm.logp(invalid_dist, value).eval()

    @pytest.mark.parametrize("n", [(10), ([10, 11]), ([[5, 6], [10, 11]])])
    @pytest.mark.parametrize(
        "p",
        [
            ([0.2, 0.3, 0.5]),
            ([[0.2, 0.3, 0.5], [0.9, 0.09, 0.01]]),
            (np.abs(np.random.randn(2, 2, 4))),
        ],
    )
    @pytest.mark.parametrize("extra_size", [(1,), (2,), (2, 3)])
    def test_multinomial_vectorized(self, n, p, extra_size):
        n = intX(np.array(n))
        p = floatX(np.array(p))
        p /= p.sum(axis=-1, keepdims=True)

        _, bcast_p = broadcast_params([n, p], ndims_params=[0, 1])
        size = extra_size + bcast_p.shape[:-1]

        mn = pm.Multinomial.dist(n=n, p=p, size=size)
        vals = mn.eval()

        assert_almost_equal(
            scipy.stats.multinomial.logpmf(vals, n, p),
            pm.logp(mn, vals).eval(),
            decimal=4,
            err_msg=f"vals={vals}",
        )

    def test_multinomial_zero_probs(self):
        # test multinomial accepts 0 probabilities / observations:
        mn = pm.Multinomial.dist(n=100, p=[0.0, 0.0, 1.0])
        assert pm.logp(mn, np.array([0, 0, 100])).eval() >= 0
        assert pm.logp(mn, np.array([50, 50, 0])).eval() == -np.inf

    @pytest.mark.parametrize("n", [2, 3])
    def test_dirichlet_multinomial(self, n):
        self.check_logp(
            DirichletMultinomial,
            Vector(Nat, n),
            {"a": Vector(Rplus, n), "n": Nat},
            dirichlet_multinomial_logpmf,
        )

    def test_dirichlet_multinomial_invalid(self):
        # Test non-scalar invalid parameters/values
        value = np.array([[1, 2, 2], [4, 0, 1]])

        invalid_dist = DirichletMultinomial.dist(n=5, a=[-1, 1, 1], size=2)
        with pytest.raises(ParameterValueError):
            pm.logp(invalid_dist, value).eval()

        value[1] -= 1
        valid_dist = DirichletMultinomial.dist(n=5, a=[1, 1, 1])
        assert np.all(np.isfinite(pm.logp(valid_dist, value).eval()) == np.array([True, False]))

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

    @pytest.mark.parametrize("n", [(10), ([10, 11]), ([[5, 6], [10, 11]])])
    @pytest.mark.parametrize(
        "a",
        [
            ([0.2, 0.3, 0.5]),
            ([[0.2, 0.3, 0.5], [0.9, 0.09, 0.01]]),
            (np.abs(np.random.randn(2, 2, 4))),
        ],
    )
    @pytest.mark.parametrize("extra_size", [(1,), (2,), (2, 3)])
    def test_dirichlet_multinomial_vectorized(self, n, a, extra_size):
        n = intX(np.array(n))
        a = floatX(np.array(a))

        _, bcast_a = broadcast_params([n, a], ndims_params=[0, 1])
        size = extra_size + bcast_a.shape[:-1]

        dm = pm.DirichletMultinomial.dist(n=n, a=a, size=size)
        vals = dm.eval()

        assert_almost_equal(
            dirichlet_multinomial_logpmf(vals, n, a),
            pm.logp(dm, vals).eval(),
            decimal=4,
            err_msg=f"vals={vals}",
        )

    @pytest.mark.parametrize(
        "value,alpha,K,logp",
        [
            (np.array([5, 4, 3, 2, 1]) / 15, 0.5, 4, 1.5126301307277439),
            (np.tile(1, 13) / 13, 2, 12, 13.980045245672827),
            (np.array([0.001] * 10 + [0.99]), 0.1, 10, -22.971662448814723),
            (np.append(0.5 ** np.arange(1, 20), 0.5**20), 5, 19, 94.20462772778092),
            (
                (np.array([[7, 5, 3, 2], [19, 17, 13, 11]]) / np.array([[17], [60]])),
                2.5,
                3,
                np.array([1.29317672, 1.50126157]),
            ),
        ],
    )
    def test_stickbreakingweights_logp(self, value, alpha, K, logp):
        with Model() as model:
            sbw = StickBreakingWeights("sbw", alpha=alpha, K=K, transform=None)
        pt = {"sbw": value}
        assert_almost_equal(
            pm.logp(sbw, value).eval(),
            logp,
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str(pt),
        )

    def test_stickbreakingweights_invalid(self):
        sbw = pm.StickBreakingWeights.dist(3.0, 3)
        sbw_wrong_K = pm.StickBreakingWeights.dist(3.0, 7)
        assert pm.logp(sbw, np.array([0.4, 0.3, 0.2, 0.15])).eval() == -np.inf
        assert pm.logp(sbw, np.array([1.1, 0.3, 0.2, 0.1])).eval() == -np.inf
        assert pm.logp(sbw, np.array([0.4, 0.3, 0.2, -0.1])).eval() == -np.inf
        assert pm.logp(sbw_wrong_K, np.array([0.4, 0.3, 0.2, 0.1])).eval() == -np.inf

    @aesara.config.change_flags(compute_test_value="raise")
    def test_categorical_bounds(self):
        with Model():
            x = Categorical("x", p=np.array([0.2, 0.3, 0.5]))
            assert np.isinf(logp(x, -1).eval())
            assert np.isinf(logp(x, 3).eval())

    @aesara.config.change_flags(compute_test_value="raise")
    @pytest.mark.parametrize(
        "p",
        [
            np.array([-0.2, 0.3, 0.5]),
            # A model where p sums to 1 but contains negative values
            np.array([-0.2, 0.7, 0.5]),
            # Hard edge case from #2082
            # Early automatic normalization of p's sum would hide the negative
            # entries if there is a single or pair number of negative values
            # and the rest are zero
            np.array([-1, -1, 0, 0]),
        ],
    )
    def test_categorical_negative_p(self, p):
        with pytest.raises(ValueError, match=f"{p}"):
            with Model():
                x = Categorical("x", p=p)

    def test_categorical_negative_p_symbolic(self):
        with pytest.raises(ParameterValueError):
            value = np.array([[1, 1, 1]])
            invalid_dist = pm.Categorical.dist(p=at.as_tensor_variable([-1, 0.5, 0.5]))
            pm.logp(invalid_dist, value).eval()

    def test_categorical_p_not_normalized_symbolic(self):
        with pytest.raises(ParameterValueError):
            value = np.array([[1, 1, 1]])
            invalid_dist = pm.Categorical.dist(p=at.as_tensor_variable([2, 2, 2]))
            pm.logp(invalid_dist, value).eval()

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_categorical(self, n):
        self.check_logp(
            Categorical,
            Domain(range(n), dtype="int64", edges=(0, n)),
            {"p": Simplex(n)},
            lambda value, p: categorical_logpdf(value, p),
        )

    def test_categorical_p_not_normalized(self):
        # test UserWarning is raised for p vals that sum to more than 1
        # and normaliation is triggered
        with pytest.warns(UserWarning, match="[5]"):
            with pm.Model() as m:
                x = pm.Categorical("x", p=[1, 1, 1, 1, 1])
        assert np.isclose(m.x.owner.inputs[3].sum().eval(), 1.0)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_orderedlogistic(self, n):
        self.check_logp(
            OrderedLogistic,
            Domain(range(n), dtype="int64", edges=(None, None)),
            {"eta": R, "cutpoints": Vector(R, n - 1)},
            lambda value, eta, cutpoints: orderedlogistic_logpdf(value, eta, cutpoints),
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_orderedprobit(self, n):
        self.check_logp(
            OrderedProbit,
            Domain(range(n), dtype="int64", edges=(None, None)),
            {"eta": Runif, "cutpoints": UnitSortedVector(n - 1)},
            lambda value, eta, cutpoints: orderedprobit_logpdf(value, eta, cutpoints),
        )

    def test_get_tau_sigma(self):
        sigma = np.array(2)
        assert_almost_equal(get_tau_sigma(sigma=sigma), [1.0 / sigma**2, sigma])

        tau = np.array(2)
        assert_almost_equal(get_tau_sigma(tau=tau), [tau, tau**-0.5])

        tau, _ = get_tau_sigma(sigma=at.constant(-2))
        with pytest.raises(ParameterValueError):
            tau.eval()

        _, sigma = get_tau_sigma(tau=at.constant(-2))
        with pytest.raises(ParameterValueError):
            sigma.eval()

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
            (-1.0, 0.0, 0.1, 0.1, -51.022349),  # Fails in previous pymc version
        ],
    )
    def test_ex_gaussian(self, value, mu, sigma, nu, logp):
        """Log probabilities calculated using the dexGAUS function from the R package gamlss.
        See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or http://www.gamlss.org/."""
        with Model() as model:
            ExGaussian("eg", mu=mu, sigma=sigma, nu=nu)
        pt = {"eg": value}
        assert_almost_equal(
            model.compile_logp()(pt),
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
            Circ,
            {"mu": R, "kappa": Rplus},
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

    def test_interpolated(self):
        for mu in R.vals:
            for sigma in Rplus.vals:
                # pylint: disable=cell-var-from-loop
                xmin = mu - 5 * sigma
                xmax = mu + 5 * sigma

                from pymc.distributions.continuous import interpolated

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

    @pytest.mark.parametrize("transform", [UNSET, None])
    def test_interpolated_transform(self, transform):
        # Issue: https://github.com/pymc-devs/pymc/issues/5048
        x_points = np.linspace(0, 10, 10)
        pdf_points = sp.norm.pdf(x_points, loc=1, scale=1)
        with pm.Model() as m:
            x = pm.Interpolated("x", x_points, pdf_points, transform=transform)

        if transform is UNSET:
            assert np.isfinite(m.compile_logp()({"x_interval__": -1.0}))
            assert np.isfinite(m.compile_logp()({"x_interval__": 11.0}))
        else:
            assert not np.isfinite(m.compile_logp()({"x": -1.0}))
            assert not np.isfinite(m.compile_logp()({"x": 11.0}))

    def test_gaussianrandomwalk(self):
        def ref_logp(value, mu, sigma, steps):
            # Relying on fact that init will be normal by default
            return (
                scipy.stats.norm.logpdf(value[0], mu, sigma)
                + scipy.stats.norm.logpdf(np.diff(value), mu, sigma).sum()
            )

        self.check_logp(
            pm.GaussianRandomWalk,
            Vector(R, 4),
            {"mu": R, "sigma": Rplus, "steps": Nat},
            ref_logp,
            decimal=select_by_precision(float64=6, float32=1),
        )


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

        assert joint_logpt(LowerNormal, -1).eval() == -np.inf
        assert joint_logpt(UpperNormal, 1).eval() == -np.inf
        assert joint_logpt(BoundedNormal, 0).eval() == -np.inf
        assert joint_logpt(BoundedNormal, 11).eval() == -np.inf

        assert joint_logpt(UnboundedNormal, 0).eval() != -np.inf
        assert joint_logpt(UnboundedNormal, 11).eval() != -np.inf
        assert joint_logpt(InfBoundedNormal, 0).eval() != -np.inf
        assert joint_logpt(InfBoundedNormal, 11).eval() != -np.inf

        value = model.rvs_to_values[LowerNormalTransform]
        assert joint_logpt(LowerNormalTransform, value).eval({value: -1}) != -np.inf
        value = model.rvs_to_values[UpperNormalTransform]
        assert joint_logpt(UpperNormalTransform, value).eval({value: 1}) != -np.inf
        value = model.rvs_to_values[BoundedNormalTransform]
        assert joint_logpt(BoundedNormalTransform, value).eval({value: 0}) != -np.inf
        assert joint_logpt(BoundedNormalTransform, value).eval({value: 11}) != -np.inf

        ref_dist = Normal.dist(mu=0, sigma=1)
        assert np.allclose(joint_logpt(UnboundedNormal, 5).eval(), joint_logpt(ref_dist, 5).eval())
        assert np.allclose(joint_logpt(LowerNormal, 5).eval(), joint_logpt(ref_dist, 5).eval())
        assert np.allclose(joint_logpt(UpperNormal, -5).eval(), joint_logpt(ref_dist, 5).eval())
        assert np.allclose(joint_logpt(BoundedNormal, 5).eval(), joint_logpt(ref_dist, 5).eval())

    def test_discrete(self):
        with Model() as model:
            dist = Poisson.dist(mu=4)
            UnboundedPoisson = Bound("unbound", dist)
            LowerPoisson = Bound("lower", dist, lower=1)
            UpperPoisson = Bound("upper", dist, upper=10)
            BoundedPoisson = Bound("bounded", dist, lower=1, upper=10)

        assert joint_logpt(LowerPoisson, 0).eval() == -np.inf
        assert joint_logpt(UpperPoisson, 11).eval() == -np.inf
        assert joint_logpt(BoundedPoisson, 0).eval() == -np.inf
        assert joint_logpt(BoundedPoisson, 11).eval() == -np.inf

        assert joint_logpt(UnboundedPoisson, 0).eval() != -np.inf
        assert joint_logpt(UnboundedPoisson, 11).eval() != -np.inf

        ref_dist = Poisson.dist(mu=4)
        assert np.allclose(joint_logpt(UnboundedPoisson, 5).eval(), joint_logpt(ref_dist, 5).eval())
        assert np.allclose(joint_logpt(LowerPoisson, 5).eval(), joint_logpt(ref_dist, 5).eval())
        assert np.allclose(joint_logpt(UpperPoisson, 5).eval(), joint_logpt(ref_dist, 5).eval())
        assert np.allclose(joint_logpt(BoundedPoisson, 5).eval(), joint_logpt(ref_dist, 5).eval())

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
                pm.Bound("bound", x, transform=pm.distributions.transforms.log)

        msg = "Given dims do not exist in model coordinates."
        with pm.Model() as m:
            x = pm.Poisson.dist(0.5)
            with pytest.raises(ValueError, match=msg):
                pm.Bound("bound", x, dims="random_dims")

        msg = "The dist x was already registered in the current model"
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

        initial_point = m.compute_initial_point()
        dist_size = initial_point["boundedsized_interval__"].shape
        dist_shape = initial_point["boundedshaped_interval__"].shape
        dist_dims = initial_point["boundeddims_interval__"].shape

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

        first, second = joint_logpt(LowerPoisson, [0, 0], sum=False)[0].eval()
        assert first == -np.inf
        assert second != -np.inf

        first, second = joint_logpt(UpperPoisson, [11, 11], sum=False)[0].eval()
        assert first != -np.inf
        assert second == -np.inf

        first, second = joint_logpt(BoundedPoisson, [1, 1], sum=False)[0].eval()
        assert first != -np.inf
        assert second == -np.inf

        first, second = joint_logpt(BoundedPoisson, [10, 10], sum=False)[0].eval()
        assert first == -np.inf
        assert second != -np.inf


class TestBoundedContinuous:
    def get_dist_params_and_interval_bounds(self, model, rv_name):
        interval_rv = model.named_vars[f"{rv_name}_interval__"]
        rv = model.named_vars[rv_name]
        dist_params = rv.owner.inputs
        lower_interval, upper_interval = interval_rv.tag.transform.args_fn(*rv.owner.inputs)
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
            (_, _, _, _, _, lower, upper),
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
            (_, _, _, _, _, lower, upper),
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
            (_, _, _, _, _, lower, upper),
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
            (_, _, _, _, _, lower, upper),
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
            pot = Potential("pot", mu**2)

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
                r"Z ~ N(f(), f())",
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
                r"$\text{Z} \sim \operatorname{N}(f(),~f())$",
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
    ologp = joint_logpt(ol, np.ones_like(obs), sum=True).eval() * loge
    clogp = joint_logpt(c, np.ones_like(obs), sum=True).eval() * loge
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
        pm.OrderedProbit("op_p", cutpoints=np.array([-2, 0, 2]), eta=0, sigma=1)
        pm.OrderedProbit("op_no_p", cutpoints=np.array([-2, 0, 2]), eta=0, sigma=1, compute_p=False)
    assert len(m.deterministics) == 1

    x = pm.OrderedProbit.dist(cutpoints=np.array([-2, 0, 2]), eta=0, sigma=1)
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
        # https://github.com/pymc-devs/pymc/issues/4499
        with pm.Model(check_bounds=False) as m:
            x = pm.Uniform("x", 0, 2, size=10, transform=None)
        assert_almost_equal(m.compile_logp()({"x": np.ones(10)}), -np.log(2) * 10)

        with pm.Model(check_bounds=False) as m:
            x = pm.DiscreteUniform("x", 0, 1, size=10)
        assert_almost_equal(m.compile_logp()({"x": np.ones(10)}), -np.log(2) * 10)

        with pm.Model(check_bounds=False) as m:
            x = pm.Constant("x", 1, size=10)
        assert_almost_equal(m.compile_logp()({"x": np.ones(10)}), 0 * 10)


def test_serialize_density_dist():
    def func(x):
        return -2 * (x**2).sum()

    def random(rng, size):
        return rng.uniform(-2, 2, size=size)

    with pm.Model():
        pm.Normal("x")
        y = pm.DensityDist("y", logp=func, random=random)
        pm.sample(draws=5, tune=1, mp_ctx="spawn")

    import cloudpickle

    cloudpickle.loads(cloudpickle.dumps(y))


def test_distinct_rvs():
    """Make sure `RandomVariable`s generated using a `Model`'s default RNG state all have distinct states."""

    with pm.Model(rng_seeder=np.random.RandomState(2023532)) as model:
        X_rv = pm.Normal("x")
        Y_rv = pm.Normal("y")

        pp_samples = pm.sample_prior_predictive(samples=2, return_inferencedata=False)

    assert X_rv.owner.inputs[0] != Y_rv.owner.inputs[0]

    assert len(model.rng_seq) == 2

    with pm.Model(rng_seeder=np.random.RandomState(2023532)):
        X_rv = pm.Normal("x")
        Y_rv = pm.Normal("y")

        pp_samples_2 = pm.sample_prior_predictive(samples=2, return_inferencedata=False)

    assert np.array_equal(pp_samples["y"], pp_samples_2["y"])


@pytest.mark.parametrize(
    "method,newcode",
    [
        ("logp", r"pm.logp\(rv, x\)"),
        ("logcdf", r"pm.logcdf\(rv, x\)"),
        ("random", r"rv.eval\(\)"),
    ],
)
def test_logp_gives_migration_instructions(method, newcode):
    rv = pm.Normal.dist()
    f = getattr(rv, method)
    with pytest.raises(AttributeError, match=rf"use `{newcode}`"):
        f()

    # A dim-induced resize of the rv created by the `.dist()` API,
    # happening in Distribution.__new__ would make us loose the monkeypatches.
    # So this triggers it to test if the monkeypatch still works.
    with pm.Model(coords={"year": [2019, 2021, 2022]}):
        rv = pm.Normal("n", dims="year")
        f = getattr(rv, method)
        with pytest.raises(AttributeError, match=rf"use `{newcode}`"):
            f()
    pass


def test_density_dist_old_api_error():
    with pm.Model():
        with pytest.raises(
            TypeError, match="The DensityDist API has changed, you are using the old API"
        ):
            pm.DensityDist("a", lambda x: x)


@pytest.mark.parametrize("size", [None, (), (2,)], ids=str)
def test_density_dist_multivariate_logp(size):
    supp_shape = 5
    with pm.Model() as model:

        def logp(value, mu):
            return pm.MvNormal.logp(value, mu, at.eye(mu.shape[0]))

        mu = pm.Normal("mu", size=supp_shape)
        a = pm.DensityDist("a", mu, logp=logp, ndims_params=[1], ndim_supp=1, size=size)
    mu_val = np.random.normal(loc=0, scale=1, size=supp_shape).astype(aesara.config.floatX)
    a_val = np.random.normal(loc=mu_val, scale=1, size=to_tuple(size) + (supp_shape,)).astype(
        aesara.config.floatX
    )
    log_densityt = joint_logpt(a, a.tag.value_var, sum=False)[0]
    assert log_densityt.eval(
        {a.tag.value_var: a_val, mu.tag.value_var: mu_val},
    ).shape == to_tuple(size)


class TestCensored:
    @pytest.mark.parametrize("censored", (False, True))
    def test_censored_workflow(self, censored):
        # Based on pymc-examples/censored_data
        rng = np.random.default_rng(1234)
        size = 500
        true_mu = 13.0
        true_sigma = 5.0

        # Set censoring limits
        low = 3.0
        high = 16.0

        # Draw censored samples
        data = rng.normal(true_mu, true_sigma, size)
        data[data <= low] = low
        data[data >= high] = high

        with pm.Model(rng_seeder=17092021) as m:
            mu = pm.Normal(
                "mu",
                mu=((high - low) / 2) + low,
                sigma=(high - low) / 2.0,
                initval="moment",
            )
            sigma = pm.HalfNormal("sigma", sigma=(high - low) / 2.0, initval="moment")
            observed = pm.Censored(
                "observed",
                pm.Normal.dist(mu=mu, sigma=sigma),
                lower=low if censored else None,
                upper=high if censored else None,
                observed=data,
            )

            prior_pred = pm.sample_prior_predictive()
            posterior = pm.sample(tune=500, draws=500)
            posterior_pred = pm.sample_posterior_predictive(posterior)

        expected = True if censored else False
        assert (9 < prior_pred.prior_predictive.mean() < 10) == expected
        assert (13 < posterior.posterior["mu"].mean() < 14) == expected
        assert (4.5 < posterior.posterior["sigma"].mean() < 5.5) == expected
        assert (12 < posterior_pred.posterior_predictive.mean() < 13) == expected

    def test_censored_invalid_dist(self):
        with pm.Model():
            invalid_dist = pm.Normal
            with pytest.raises(
                ValueError,
                match=r"Censoring dist must be a distribution created via the",
            ):
                x = pm.Censored("x", invalid_dist, lower=None, upper=None)

        with pm.Model():
            mv_dist = pm.Dirichlet.dist(a=[1, 1, 1])
            with pytest.raises(
                NotImplementedError,
                match="Censoring of multivariate distributions has not been implemented yet",
            ):
                x = pm.Censored("x", mv_dist, lower=None, upper=None)

        with pm.Model():
            registered_dist = pm.Normal("dist")
            with pytest.raises(
                ValueError,
                match="The dist dist was already registered in the current model",
            ):
                x = pm.Censored("x", registered_dist, lower=None, upper=None)

    def test_change_size(self):
        base_dist = pm.Censored.dist(pm.Normal.dist(), -1, 1, size=(3, 2))

        new_dist = pm.Censored.change_size(base_dist, (4,))
        assert new_dist.eval().shape == (4,)

        new_dist = pm.Censored.change_size(base_dist, (4,), expand=True)
        assert new_dist.eval().shape == (4, 3, 2)


class TestLKJCholeskCov:
    def test_dist(self):
        sd_dist = pm.Exponential.dist(1, size=(10, 3))
        x = pm.LKJCholeskyCov.dist(n=3, eta=1, sd_dist=sd_dist, size=10, compute_corr=False)
        assert x.eval().shape == (10, 6)

        sd_dist = pm.Exponential.dist(1, size=3)
        chol, corr, stds = pm.LKJCholeskyCov.dist(n=3, eta=1, sd_dist=sd_dist)
        assert chol.eval().shape == (3, 3)
        assert corr.eval().shape == (3, 3)
        assert stds.eval().shape == (3,)

    def test_sd_dist_distribution(self):
        with pm.Model() as m:
            sd_dist = at.constant([1, 2, 3])
            with pytest.raises(TypeError, match="^sd_dist must be a scalar or vector distribution"):
                x = pm.LKJCholeskyCov("x", n=3, eta=1, sd_dist=sd_dist)

    def test_sd_dist_registered(self):
        with pm.Model() as m:
            sd_dist = pm.Exponential("sd_dist", 1, size=3)
            with pytest.raises(
                ValueError, match="The dist sd_dist was already registered in the current model"
            ):
                x = pm.LKJCholeskyCov("x", n=3, eta=1, sd_dist=sd_dist)

    def test_no_warning_logp(self):
        # Check that calling logp of a model with LKJCholeskyCov does not issue any warnings
        # due to the RandomVariable in the graph
        with pm.Model() as m:
            sd_dist = pm.Exponential.dist(1, size=3)
            x = pm.LKJCholeskyCov("x", n=3, eta=1, sd_dist=sd_dist)
        with pytest.warns(None) as record:
            m.logpt()
        assert not record

    @pytest.mark.parametrize(
        "sd_dist",
        [
            pm.Exponential.dist(1),
            pm.MvNormal.dist(np.ones(3), np.eye(3)),
        ],
    )
    def test_sd_dist_automatically_resized(self, sd_dist):
        x = pm.LKJCholeskyCov.dist(n=3, eta=1, sd_dist=sd_dist, size=10, compute_corr=False)
        resized_sd_dist = x.owner.inputs[-1]
        assert resized_sd_dist.eval().shape == (10, 3)
        # LKJCov has support shape `(n * (n+1)) // 2`
        assert x.eval().shape == (10, 6)


@pytest.mark.parametrize(
    "dist, non_psi_args",
    [
        (pm.ZeroInflatedPoisson.dist, (2,)),
        (pm.ZeroInflatedBinomial.dist, (2, 0.5)),
        (pm.ZeroInflatedNegativeBinomial.dist, (2, 2)),
    ],
)
def test_zero_inflated_dists_dtype_and_broadcast(dist, non_psi_args):
    x = dist([0.5, 0.5, 0.5], *non_psi_args)
    assert x.dtype in discrete_types
    assert x.eval().shape == (3,)
