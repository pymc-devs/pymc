"""Tests for reparameterization rewrites.

Two checks per distribution:
  - Distributional correctness: empirical median and IQR of reparameterized
    samples match the analytical distribution within Monte Carlo tolerance.
  - Gradient correctness: autodiff gradient of mean(samples) with respect to
    each distribution parameter matches a finite-difference estimate of the
    analytical mean from scipy.

The robust statistics (median/IQR) are used in place of mean/std so the same
check works for heavy-tailed distributions (Cauchy, Pareto, etc.).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats as st

from pytensor.tensor.random.rewriting.basic import local_rv_size_lift

from pymc.distributions.rewrites import reparameterization as rp

SEED = 42
N = 30_000
FD_STEP = 1e-3
DIST_RTOL = 0.05
DIST_ATOL = 0.05
GRAD_RTOL = 0.05
GRAD_ATOL = 0.02


_NEEDS_SIZE_LIFT = {
    rp.gamma_reparametrization,
    rp.inv_gamma_reparametrization,
    rp.beta_reparametrization,
    rp.dirichlet_reparametrization,
    rp.wald_reparametrization,
}


def reparameterize(rewriter, rv):
    if rewriter in _NEEDS_SIZE_LIFT:
        lifted = local_rv_size_lift.fn(None, rv.owner)
        node = lifted[1].owner if lifted else rv.owner
    else:
        node = rv.owner
    return rewriter.fn(None, node)


def make_rng(seed=SEED):
    return pytensor.shared(np.random.default_rng(seed))


def empirical_median_iqr(samples):
    q25, q50, q75 = np.percentile(samples, [25, 50, 75])
    return q50, q75 - q25


def reference_median_iqr(scipy_dist):
    return scipy_dist.median(), scipy_dist.ppf(0.75) - scipy_dist.ppf(0.25)


@dataclass(frozen=True)
class Case:
    id: str
    rewriter: Callable
    build_rv: Callable                   # (rng, size, *param_values) -> RV
    scipy_factory: Callable | None       # (*param_values) -> scipy frozen dist; None for multivariate
    param_values: tuple
    diff_param_idx: tuple = ()           # which parameters to autodiff against; () skips the gradient test


def _normal_rv(rng, size, loc, scale):
    return pt.random.normal(loc, scale, size=size, rng=rng)


def _cauchy_rv(rng, size, loc, scale):
    return pt.random.cauchy(loc, scale, size=size, rng=rng)


def _gumbel_rv(rng, size, loc, scale):
    return pt.random.gumbel(loc, scale, size=size, rng=rng)


def _halfcauchy_rv(rng, size, loc, scale):
    return pt.random.halfcauchy(loc, scale, size=size, rng=rng)


def _halfnormal_rv(rng, size, loc, scale):
    return pt.random.halfnormal(loc, scale, size=size, rng=rng)


def _laplace_rv(rng, size, loc, scale):
    return pt.random.laplace(loc, scale, size=size, rng=rng)


def _logistic_rv(rng, size, loc, scale):
    return pt.random.logistic(loc, scale, size=size, rng=rng)


def _exponential_rv(rng, size, scale):
    return pt.random.exponential(scale, size=size, rng=rng)


def _lognormal_rv(rng, size, mean, sigma):
    return pt.random.lognormal(mean, sigma, size=size, rng=rng)


def _gamma_rv(rng, size, shape, scale):
    return pt.random.gamma(shape, scale=scale, size=size, rng=rng)


def _beta_rv(rng, size, alpha, beta_):
    return pt.random.beta(alpha, beta_, size=size, rng=rng)


def _kumaraswamy_rv(rng, size, a, b):
    with pm.Model() as model:
        pm.Kumaraswamy("x", a=a, b=b, size=size)
    return model.free_RVs[0]


def _pareto_rv(rng, size, b, scale):
    return pt.random.pareto(b, scale, size=size, rng=rng)


def _studentt_rv(rng, size, df, loc, scale):
    return pt.random.t(df, loc, scale, size=size, rng=rng)


def _triangular_rv(rng, size, left, mode, right):
    return pt.random.triangular(left, mode, right, size=size, rng=rng)


def _uniform_rv(rng, size, low, high):
    return pt.random.uniform(low, high, size=size, rng=rng)


def _wald_rv(rng, size, mean, scale):
    return pt.random.wald(mean, scale, size=size, rng=rng)


def _weibull_rv(rng, size, shape):
    return pt.random.weibull(shape, size=size, rng=rng)


def _bernoulli_rv(rng, size, p):
    return pt.random.bernoulli(p, size=size, rng=rng)


def _categorical_rv(rng, size, p):
    return pt.random.categorical(p, size=size, rng=rng)


def _geometric_rv(rng, size, p):
    return pt.random.geometric(p, size=size, rng=rng)


CONTINUOUS_CASES = [
    Case("normal", rp.loc_scale_reparametrization, _normal_rv,
         lambda loc, scale: st.norm(loc=loc, scale=scale), (2.0, 1.5), (0, 1)),
    Case("cauchy", rp.loc_scale_reparametrization, _cauchy_rv,
         lambda loc, scale: st.cauchy(loc=loc, scale=scale), (1.0, 0.7), ()),
    Case("gumbel", rp.loc_scale_reparametrization, _gumbel_rv,
         lambda loc, scale: st.gumbel_r(loc=loc, scale=scale), (0.5, 1.2), (0, 1)),
    Case("halfcauchy", rp.loc_scale_reparametrization, _halfcauchy_rv,
         lambda loc, scale: st.halfcauchy(loc=loc, scale=scale), (0.0, 1.5), ()),
    Case("halfnormal", rp.loc_scale_reparametrization, _halfnormal_rv,
         lambda loc, scale: st.halfnorm(loc=loc, scale=scale), (0.0, 1.5), (1,)),
    Case("laplace", rp.loc_scale_reparametrization, _laplace_rv,
         lambda loc, scale: st.laplace(loc=loc, scale=scale), (1.0, 0.8), (0, 1)),
    Case("logistic", rp.loc_scale_reparametrization, _logistic_rv,
         lambda loc, scale: st.logistic(loc=loc, scale=scale), (1.0, 0.8), (0, 1)),
    Case("exponential", rp.scale_reparametrization, _exponential_rv,
         lambda scale: st.expon(scale=scale), (1.5,), (0,)),
    Case("lognormal", rp.log_normal_reparametrization, _lognormal_rv,
         lambda mean, sigma: st.lognorm(s=sigma, scale=np.exp(mean)), (0.5, 0.4), (0, 1)),
    Case("gamma", rp.gamma_reparametrization, _gamma_rv,
         lambda shape, scale: st.gamma(a=shape, scale=scale), (2.5, 1.5), (0, 1)),
    Case("beta", rp.beta_reparametrization, _beta_rv,
         lambda a, b: st.beta(a=a, b=b), (2.0, 5.0), (0, 1)),
    Case("kumaraswamy", rp.kumaraswamy_reparametrization, _kumaraswamy_rv,
         None, (2.0, 5.0), (0, 1)),
    Case("pareto", rp.pareto_reparametrization, _pareto_rv,
         lambda b, scale: st.pareto(b=b, scale=scale), (3.0, 1.0), (0, 1)),
    Case("studentt", rp.student_t_reparametrization, _studentt_rv,
         lambda df, loc, scale: st.t(df=df, loc=loc, scale=scale), (5.0, 0.5, 1.2), (1, 2)),
    Case("triangular", rp.triangular_reparametrization, _triangular_rv,
         lambda left, mode, right: st.triang(c=(mode - left) / (right - left), loc=left, scale=right - left),
         (0.0, 0.3, 1.0), (0, 1, 2)),
    Case("uniform", rp.uniform_reparametrization, _uniform_rv,
         lambda low, high: st.uniform(loc=low, scale=high - low), (-1.0, 2.0), (0, 1)),
    Case("wald", rp.wald_reparametrization, _wald_rv,
         lambda mean, scale: st.invgauss(mu=mean / scale, scale=scale), (1.0, 2.0), (0, 1)),
    Case("weibull", rp.weibull_reparametrization, _weibull_rv,
         lambda shape: st.weibull_min(c=shape), (1.5,), (0,)),
]


DISCRETE_CASES = [
    Case("bernoulli", rp.bernoulli_reparametrization, _bernoulli_rv,
         lambda p: st.bernoulli(p=p), (0.3,), ()),
    Case("categorical", rp.categorical_reparametrization, _categorical_rv,
         lambda p: st.rv_discrete(values=(np.arange(len(p)), p)), (np.array([0.1, 0.3, 0.6]),), ()),
    Case("geometric", rp.geometric_reparametrization, _geometric_rv,
         lambda p: st.geom(p=p), (0.4,), ()),
]


def _kumaraswamy_median_iqr(a, b):
    median = (1 - 0.5 ** (1 / b)) ** (1 / a)
    q25 = (1 - 0.75 ** (1 / b)) ** (1 / a)
    q75 = (1 - 0.25 ** (1 / b)) ** (1 / a)
    return median, q75 - q25


@pytest.mark.parametrize("case", CONTINUOUS_CASES, ids=[c.id for c in CONTINUOUS_CASES])
def test_continuous_distribution(case):
    rng = make_rng()
    rv = case.build_rv(rng, N, *case.param_values)
    new = reparameterize(case.rewriter, rv)
    samples = pytensor.function([], new)()

    if case.id == "kumaraswamy":
        ref_med, ref_iqr = _kumaraswamy_median_iqr(*case.param_values)
    else:
        ref_med, ref_iqr = reference_median_iqr(case.scipy_factory(*case.param_values))

    emp_med, emp_iqr = empirical_median_iqr(samples)
    np.testing.assert_allclose(emp_med, ref_med, rtol=DIST_RTOL, atol=DIST_ATOL)
    np.testing.assert_allclose(emp_iqr, ref_iqr, rtol=DIST_RTOL, atol=DIST_ATOL)


_GRADIENT_XFAIL = {}


def _gradient_marks(case):
    reason = _GRADIENT_XFAIL.get(case.id)
    return [pytest.mark.xfail(reason=reason, strict=True)] if reason else []


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, marks=_gradient_marks(c), id=c.id)
        for c in CONTINUOUS_CASES
        if c.diff_param_idx and c.scipy_factory is not None
    ],
)
def test_continuous_gradient(case):
    rng = make_rng()
    sym_params = [pt.scalar(f"p{i}") for i in range(len(case.param_values))]
    rv = case.build_rv(rng, N, *sym_params)
    new = reparameterize(case.rewriter, rv)

    diff_params = [sym_params[i] for i in case.diff_param_idx]
    mean_expr = new.mean()
    grads = pt.grad(mean_expr, diff_params)
    f = pytensor.function(sym_params, [mean_expr] + grads)

    result = f(*case.param_values)
    autodiff_grads = result[1:]

    fd_grads = []
    for i in case.diff_param_idx:
        plus = list(case.param_values)
        minus = list(case.param_values)
        plus[i] += FD_STEP
        minus[i] -= FD_STEP
        m_plus = case.scipy_factory(*plus).mean()
        m_minus = case.scipy_factory(*minus).mean()
        fd_grads.append((m_plus - m_minus) / (2 * FD_STEP))

    np.testing.assert_allclose(autodiff_grads, fd_grads, rtol=GRAD_RTOL, atol=GRAD_ATOL)


@pytest.mark.parametrize("case", DISCRETE_CASES, ids=[c.id for c in DISCRETE_CASES])
def test_discrete_distribution(case):
    rng = make_rng()
    rv = case.build_rv(rng, N, *case.param_values)
    new = reparameterize(case.rewriter, rv)
    samples = pytensor.function([], new)()

    if case.id == "categorical":
        p = case.param_values[0]
        emp = np.bincount(samples.astype(int), minlength=len(p)) / len(samples)
        np.testing.assert_allclose(emp, p, atol=0.02)
    else:
        ref = case.scipy_factory(*case.param_values)
        np.testing.assert_allclose(samples.mean(), ref.mean(), rtol=DIST_RTOL, atol=DIST_ATOL)
        np.testing.assert_allclose(samples.std(), ref.std(), rtol=DIST_RTOL, atol=DIST_ATOL)


def test_dirichlet_distribution():
    rng = make_rng()
    alpha = np.array([2.0, 5.0, 3.0])
    rv = pt.random.dirichlet(alpha, size=(N,), rng=rng)
    new = reparameterize(rp.dirichlet_reparametrization, rv)
    samples = pytensor.function([], new)()

    expected_mean = alpha / alpha.sum()
    np.testing.assert_allclose(samples.mean(axis=0), expected_mean, rtol=DIST_RTOL, atol=DIST_ATOL)
    np.testing.assert_allclose(samples.sum(axis=-1), 1.0, atol=1e-6)


def test_mvnormal_distribution():
    rng = make_rng()
    mean = np.array([1.0, -0.5])
    cov = np.array([[1.0, 0.4], [0.4, 0.7]])
    rv = pt.random.multivariate_normal(mean, cov, size=(N,), rng=rng)
    new = reparameterize(rp.mv_normal_reparametrization, rv)
    samples = pytensor.function([], new)()

    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.05)
    emp_cov = np.cov(samples, rowvar=False)
    np.testing.assert_allclose(emp_cov, cov, atol=0.1)


def test_invgamma_distribution():
    rng = make_rng()
    shape, scale = 4.0, 2.0
    with pm.Model() as model:
        pm.InverseGamma("x", alpha=shape, beta=scale, size=N)
    rv = model.free_RVs[0]
    new = reparameterize(rp.inv_gamma_reparametrization, rv)
    samples = pytensor.function([], new)()

    ref = st.invgamma(a=shape, scale=scale)
    emp_med, emp_iqr = empirical_median_iqr(samples)
    ref_med, ref_iqr = reference_median_iqr(ref)
    np.testing.assert_allclose(emp_med, ref_med, rtol=DIST_RTOL, atol=DIST_ATOL)
    np.testing.assert_allclose(emp_iqr, ref_iqr, rtol=DIST_RTOL, atol=DIST_ATOL)
