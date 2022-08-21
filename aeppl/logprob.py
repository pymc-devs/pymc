from functools import singledispatch
from typing import Tuple

import aesara.tensor as at
import aesara.tensor.random.basic as arb
import numpy as np
from aesara.graph.op import Op
from aesara.raise_op import CheckAndRaise
from aesara.tensor.slinalg import Cholesky, solve_lower_triangular
from aesara.tensor.var import TensorVariable

from aeppl.dists import DiracDelta, DiscreteMarkovChainFactory


class ParameterValueError(ValueError):
    """Exception for invalid parameters values in logprob graphs"""


class CheckParameterValue(CheckAndRaise):
    """Implements a parameter value check in a logprob graph.

    Raises `ParameterValueError` if the check is not True.
    """

    def __init__(self, msg=""):
        super().__init__(ParameterValueError, msg)

    def __str__(self):
        return f"Check{{{self.msg}}}"


cholesky = Cholesky(lower=True, on_error="nan")


def betaln(x, y):
    return at.gammaln(x) + at.gammaln(y) - at.gammaln(x + y)


def binomln(n, k):
    return at.gammaln(n + 1) - at.gammaln(k + 1) - at.gammaln(n - k + 1)


def xlogy0(m, x):
    # TODO: This should probably be a basic Aesara stabilization
    return at.switch(at.eq(x, 0), at.switch(at.eq(m, 0), 0.0, -np.inf), m * at.log(x))


def logprob(rv_var, *rv_values, **kwargs):
    """Create a graph for the log-probability of a ``RandomVariable``."""
    logprob = _logprob(rv_var.owner.op, rv_values, *rv_var.owner.inputs, **kwargs)

    for rv_var in rv_values:
        if rv_var.name:
            logprob.name = f"{rv_var.name}_logprob"

    return logprob


def logcdf(rv_var, rv_value, **kwargs):
    """Create a graph for the logcdf of a ``RandomVariable``."""
    logcdf = _logcdf(
        rv_var.owner.op, rv_value, *rv_var.owner.inputs, name=rv_var.name, **kwargs
    )

    if rv_var.name:
        logcdf.name = f"{rv_var.name}_logcdf"

    return logcdf


def icdf(rv, value, **kwargs):
    """Create a graph for the inverse CDF of a `RandomVariable`."""
    rv_icdf = _icdf(rv.owner.op, value, *rv.owner.inputs, **kwargs)
    if rv.name:
        rv_icdf.name = f"{rv.name}_icdf"
    return rv_icdf


@singledispatch
def _logprob(
    op: Op,
    values: Tuple[TensorVariable],
    *inputs: TensorVariable,
    **kwargs,
):
    """Create a graph for the log-density/mass of a ``RandomVariable``.

    This function dispatches on the type of ``op``, which should be a subclass
    of ``RandomVariable``.  If you want to implement new density/mass graphs
    for a ``RandomVariable``, register a new function on this dispatcher.

    """
    raise NotImplementedError(f"Logprob method not implemented for {op}")


@singledispatch
def _logcdf(
    op: Op,
    value: TensorVariable,
    *inputs: TensorVariable,
    **kwargs,
):
    """Create a graph for the logcdf of a ``RandomVariable``.

    This function dispatches on the type of ``op``, which should be a subclass
    of ``RandomVariable``.  If you want to implement new logcdf graphs
    for a ``RandomVariable``, register a new function on this dispatcher.
    """
    raise NotImplementedError(f"Logcdf method not implemented for {op}")


@singledispatch
def _icdf(
    op: Op,
    value: TensorVariable,
    *inputs: TensorVariable,
    **kwargs,
):
    """Create a graph for the inverse CDF of a `RandomVariable`.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.
    """
    raise NotImplementedError(f"icdf not implemented for {op}")


@_logprob.register(arb.UniformRV)
def uniform_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    lower, upper = inputs[3:]
    return at.switch(
        at.bitwise_and(at.ge(value, lower), at.le(value, upper)),
        at.fill(value, -at.log(upper - lower)),
        -np.inf,
    )


@_logcdf.register(arb.UniformRV)
def uniform_logcdf(op, value, *inputs, **kwargs):
    lower, upper = inputs[3:]

    res = at.switch(
        at.lt(value, lower),
        -np.inf,
        at.switch(
            at.lt(value, upper),
            at.log(value - lower) - at.log(upper - lower),
            0,
        ),
    )

    res = CheckParameterValue("lower <= upper")(res, at.all(at.le(lower, upper)))
    return res


@_logprob.register(arb.NormalRV)
def normal_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, sigma = inputs[3:]
    res = (
        -0.5 * at.pow((value - mu) / sigma, 2)
        - at.log(at.sqrt(2.0 * np.pi))
        - at.log(sigma)
    )
    res = CheckParameterValue("sigma > 0")(res, at.all(at.gt(sigma, 0.0)))
    return res


@_logcdf.register(arb.NormalRV)
def normal_logcdf(op, value, *inputs, **kwargs):
    mu, sigma = inputs[3:]

    z = (value - mu) / sigma
    res = at.switch(
        at.lt(z, -1.0),
        at.log(at.erfcx(-z / at.sqrt(2.0)) / 2.0) - at.sqr(z) / 2.0,
        at.log1p(-at.erfc(z / at.sqrt(2.0)) / 2.0),
    )

    res = CheckParameterValue("sigma > 0")(res, at.all(at.gt(sigma, 0.0)))
    return res


@_icdf.register(arb.NormalRV)
def normal_icdf(op, value, *inputs, **kwargs):
    loc, scale = inputs[3:]
    return loc + scale * -np.sqrt(2.0) * at.erfcinv(2 * value)


@_logprob.register(arb.HalfNormalRV)
def halfnormal_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    loc, sigma = inputs[3:]
    res = (
        -0.5 * at.pow((value - loc) / sigma, 2)
        + at.log(at.sqrt(2.0 / np.pi))
        - at.log(sigma)
    )
    res = at.switch(at.ge(value, loc), res, -np.inf)
    res = CheckParameterValue("sigma > 0")(res, at.all(at.gt(sigma, 0.0)))
    return res


@_logprob.register(arb.BetaRV)
def beta_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    alpha, beta = inputs[3:]
    res = (
        at.switch(at.eq(alpha, 1.0), 0.0, (alpha - 1.0) * at.log(value))
        + at.switch(at.eq(beta, 1.0), 0.0, (beta - 1.0) * at.log1p(-value))
        - (at.gammaln(alpha) + at.gammaln(beta) - at.gammaln(alpha + beta))
    )
    res = at.switch(at.bitwise_and(at.ge(value, 0.0), at.le(value, 1.0)), res, -np.inf)
    res = CheckParameterValue("0 <= value <= 1, alpha > 0, beta > 0")(
        res, at.all(at.gt(alpha, 0.0)), at.all(at.gt(beta, 0.0))
    )
    return res


@_logprob.register(arb.ExponentialRV)
def exponential_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (mu,) = inputs[3:]
    res = -at.log(mu) - value / mu
    res = at.switch(at.ge(value, 0.0), res, -np.inf)
    res = CheckParameterValue("mu > 0")(res, at.all(at.gt(mu, 0.0)))
    return res


@_logprob.register(arb.LaplaceRV)
def laplace_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, b = inputs[3:]
    res = -at.log(2 * b) - at.abs_(value - mu) / b
    res = CheckParameterValue("b > 0")(res, at.all(at.gt(b, 0.0)))
    return res


@_logprob.register(arb.LogNormalRV)
def lognormal_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, sigma = inputs[3:]
    res = (
        -0.5 * at.pow((at.log(value) - mu) / sigma, 2)
        - 0.5 * at.log(2.0 * np.pi)
        - at.log(sigma)
        - at.log(value)
    )
    res = at.switch(at.gt(value, 0.0), res, -np.inf)
    res = CheckParameterValue("sigma > 0")(res, at.all(at.gt(sigma, 0)))
    return res


@_logprob.register(arb.ParetoRV)
def pareto_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    alpha, m = inputs[3:]
    res = at.log(alpha) + xlogy0(alpha, m) - xlogy0(alpha + 1.0, value)
    res = at.switch(at.ge(value, m), res, -np.inf)
    res = CheckParameterValue("alpha > 0, m > 0")(
        res, at.all(at.gt(alpha, 0.0)), at.all(at.gt(m, 0.0))
    )
    return res


@_logprob.register(arb.CauchyRV)
def cauchy_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    alpha, beta = inputs[3:]
    res = -at.log(np.pi) - at.log(beta) - at.log1p(at.pow((value - alpha) / beta, 2))
    res = CheckParameterValue("beta > 0")(res, at.all(at.gt(beta, 0.0)))
    return res


@_logprob.register(arb.HalfCauchyRV)
def halfcauchy_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    res = at.log(2) + cauchy_logprob(op, values, *inputs, **kwargs)
    loc, _ = inputs[3:]
    res = at.switch(at.ge(value, loc), res, -np.inf)
    return res


@_logprob.register(arb.GammaRV)
def gamma_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    alpha, inv_beta = inputs[3:]
    beta = at.reciprocal(inv_beta)
    res = (
        -at.gammaln(alpha)
        + xlogy0(alpha, beta)
        - beta * value
        + xlogy0(alpha - 1, value)
    )
    res = at.switch(at.ge(value, 0.0), res, -np.inf)
    res = CheckParameterValue("alpha > 0, beta > 0")(
        res, at.all(at.gt(alpha, 0.0)), at.all(at.gt(beta, 0.0))
    )
    return res


@_logprob.register(arb.InvGammaRV)
def invgamma_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    alpha, beta = inputs[3:]
    res = (
        -at.gammaln(alpha)
        + xlogy0(alpha, beta)
        - beta / value
        + xlogy0(-alpha - 1, value)
    )
    res = at.switch(at.ge(value, 0.0), res, -np.inf)
    res = CheckParameterValue("alpha > 0, beta > 0")(
        res, at.all(at.gt(alpha, 0.0)), at.all(at.gt(beta, 0.0))
    )
    return res


@_logprob.register(arb.ChiSquareRV)
def chisquare_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (nu,) = inputs[3:]
    res = gamma_logprob(op, values, *(*inputs[:3], nu / 2, 2))
    return res


@_logprob.register(arb.WaldRV)
def wald_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, scale = inputs[3:]
    res = (
        0.5 * at.log(scale / (2.0 * np.pi))
        - 1.5 * at.log(value)
        - 0.5 * scale / value * ((value - mu) / mu) ** 2
    )

    res = at.switch(at.gt(value, 0.0), res, -np.inf)
    res = CheckParameterValue("mu > 0, scale > 0")(
        res, at.all(at.gt(mu, 0.0)), at.all(at.gt(scale, 0.0))
    )
    return res


@_logprob.register(arb.WeibullRV)
def weibull_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (c,) = inputs[3:]
    res = at.log(c) + (c - 1.0) * at.log(value) - at.pow(value, c)
    res = at.switch(at.ge(value, 0.0), res, -np.inf)
    res = CheckParameterValue("c > 0")(res, at.all(at.gt(c, 0.0)))
    return res


@_logprob.register(arb.VonMisesRV)
def vonmises_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, kappa = inputs[3:]
    res = kappa * at.cos(mu - value) - at.log(2 * np.pi) - at.log(at.i0(kappa))
    res = at.switch(
        at.bitwise_and(at.ge(value, -np.pi), at.le(value, np.pi)), res, -np.inf
    )
    res = CheckParameterValue("kappa > 0")(res, at.all(at.gt(kappa, 0.0)))
    return res


@_logprob.register(arb.TriangularRV)
def triangular_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    lower, c, upper = inputs[3:]
    res = at.switch(
        at.lt(value, c),
        at.log(2 * (value - lower) / ((upper - lower) * (c - lower))),
        at.log(2 * (upper - value) / ((upper - lower) * (upper - c))),
    )
    res = at.switch(
        at.bitwise_and(at.le(lower, value), at.le(value, upper)), res, -np.inf
    )
    res = CheckParameterValue("lower <= c, c <= upper")(
        res, at.all(at.le(lower, c)), at.all(at.le(c, upper))
    )
    return res


@_logprob.register(arb.GumbelRV)
def gumbel_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, beta = inputs[3:]
    z = (value - mu) / beta
    res = -z - at.exp(-z) - at.log(beta)
    res = CheckParameterValue("0 < beta")(res, at.all(at.lt(0.0, beta)))
    return res


@_logprob.register(arb.LogisticRV)
def logistic_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, s = inputs[3:]
    z = (value - mu) / s
    res = -z - at.log(s) - 2.0 * at.log1p(at.exp(-z))
    res = CheckParameterValue("0 < s")(res, at.all(at.lt(0.0, s)))
    return res


@_logprob.register(arb.BinomialRV)
def binomial_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    n, p = inputs[3:]
    res = binomln(n, value) + xlogy0(value, p) + xlogy0(n - value, 1.0 - p)
    res = at.switch(at.bitwise_and(at.le(0, value), at.le(value, n)), res, -np.inf)
    res = CheckParameterValue("0 <= p, p <= 1")(
        res, at.all(at.le(0.0, p)), at.all(at.le(p, 1.0))
    )
    return res


@_logprob.register(arb.BetaBinomialRV)
def betabinomial_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    n, alpha, beta = inputs[3:]
    res = (
        binomln(n, value)
        + betaln(value + alpha, n - value + beta)
        - betaln(alpha, beta)
    )
    res = at.switch(at.bitwise_and(at.le(0, value), at.le(value, n)), res, -np.inf)
    res = CheckParameterValue("0 < alpha, 0 < beta")(
        res, at.all(at.lt(0.0, alpha)), at.all(at.lt(0.0, beta))
    )
    return res


@_logprob.register(arb.BernoulliRV)
def bernoulli_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (p,) = inputs[3:]
    res = at.switch(value, at.log(p), at.log(1.0 - p))
    res = at.switch(at.bitwise_and(at.le(0, value), at.le(value, 1)), res, -np.inf)
    res = CheckParameterValue("0 <= p <= 1")(
        res, at.all(at.le(0.0, p)), at.all(at.le(p, 1.0))
    )
    return res


@_logprob.register(arb.PoissonRV)
def poisson_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (mu,) = inputs[3:]
    res = xlogy0(value, mu) - at.gammaln(value + 1) - mu
    res = at.switch(at.le(0, value), res, -np.inf)
    res = CheckParameterValue("0 <= mu")(res, at.all(at.le(0.0, mu)))
    res = at.switch(at.bitwise_and(at.eq(mu, 0.0), at.eq(value, 0.0)), 0.0, res)
    return res


@_logcdf.register(arb.PoissonRV)
def poisson_logcdf(op, value, *inputs, **kwargs):
    (mu,) = inputs[3:]
    value = at.floor(value)
    res = at.log(at.gammaincc(value + 1, mu))
    res = at.switch(at.le(0, value), res, -np.inf)
    res = CheckParameterValue("0 <= mu")(res, at.all(at.le(0.0, mu)))
    return res


@_logprob.register(arb.NegBinomialRV)
def nbinom_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    n, p = inputs[3:]
    mu = n * (1 - p) / p
    res = (
        binomln(value + n - 1, value)
        + xlogy0(value, mu / (mu + n))
        + xlogy0(n, n / (mu + n))
    )
    res = at.switch(at.le(0, value), res, -np.inf)
    res = CheckParameterValue("0 < mu, 0 < n")(
        res, at.all(at.lt(0.0, mu)), at.all(at.lt(0.0, n))
    )
    res = at.switch(at.gt(n, 1e10), poisson_logprob(op, values, *inputs[:3], mu), res)
    return res


@_logprob.register(arb.GeometricRV)
def geometric_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (p,) = inputs[3:]
    res = at.log(p) + xlogy0(value - 1, 1 - p)
    res = at.switch(at.le(1, value), res, -np.inf)
    res = CheckParameterValue("0 <= p <= 1")(
        res, at.all(at.le(0.0, p)), at.all(at.ge(1.0, p))
    )
    return res


@_logcdf.register(arb.GeometricRV)
def geometric_logcdf(op, value, *inputs, **kwargs):
    (p,) = inputs[3:]
    res = at.switch(at.le(value, 0), -np.inf, at.log1mexp(at.log1p(-p) * value))
    res = CheckParameterValue("0 <= p <= 1")(
        res, at.all(at.le(0.0, p)), at.all(at.ge(1.0, p))
    )
    return res


@_icdf.register(arb.GeometricRV)
def geometric_icdf(op, value, *inputs, **kwargs):
    (p,) = inputs[3:]
    return at.ceil(at.log1p(-value) / at.log1p(-p)).astype(op.dtype)


@_logprob.register(arb.HyperGeometricRV)
def hypergeometric_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    good, bad, n = inputs[3:]
    total = good + bad
    res = (
        betaln(good + 1, 1)
        + betaln(bad + 1, 1)
        + betaln(total - n + 1, n + 1)
        - betaln(value + 1, good - value + 1)
        - betaln(n - value + 1, bad - n + value + 1)
        - betaln(total + 1, 1)
    )
    lower = at.switch(at.gt(n - total + good, 0), n - total + good, 0)
    upper = at.switch(at.lt(good, n), good, n)
    res = at.switch(
        at.bitwise_and(at.le(lower, value), at.le(value, upper)), res, -np.inf
    )
    return res


@_logprob.register(arb.CategoricalRV)
def categorical_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (p,) = inputs[3:]

    p = p / at.sum(p, axis=-1, keepdims=True)

    if p.ndim > 1:
        if p.ndim > value.ndim:
            value = at.shape_padleft(value, p.ndim - value.ndim)
        elif p.ndim < value.ndim:
            p = at.shape_padleft(p, value.ndim - p.ndim)

        pattern = (p.ndim - 1,) + tuple(range(p.ndim - 1))
        res = at.log(
            at.take_along_axis(
                p.dimshuffle(pattern),
                value,
            )
        )
        # FIXME: `take_along_axis` drops a broadcastable dimension
        # when `value.broadcastable == p.broadcastable == (True, True, False)`.
    else:
        res = at.log(p[value])

    res = at.switch(
        at.bitwise_and(at.le(0, value), at.lt(value, at.shape(p)[-1])), res, -np.inf
    )
    res = CheckParameterValue("0 <= p <= 1")(
        res, at.all(at.ge(p, 0.0)), at.all(at.le(p, 1.0))
    )
    return res


@_logprob.register(arb.MvNormalRV)
def mvnormal_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, cov = inputs[3:]

    r = value - mu
    cov_chol = cholesky(cov)

    cov_chol_diag = at.diag(cov_chol)

    # TODO: Tag these matrices as positive definite when they're created
    # Use pseudo-determinant instead.  E.g. from SciPy,
    # s, u = eigh(cov)
    # factor = {'f': 1E3, 'd': 1E6}
    # t = s.numpy_dtype.char.lower()
    # cond = factor[t] * np.finfo(t).eps
    # eps = cond * at.max(at.abs_(s))
    # n = s[at.gt(s, eps)]

    all_pos_definite = at.all(at.gt(cov_chol_diag, 0))
    cov_chol = at.switch(all_pos_definite, cov_chol, 1)

    z_T = solve_lower_triangular(cov_chol, r.T).T
    quaddist = at.pow(z_T, 2).sum(axis=-1)

    logdet = at.sum(at.log(cov_chol_diag))

    n = value.shape[-1]
    res = -0.5 * n * at.log(2 * np.pi) - 0.5 * quaddist - logdet
    res = CheckParameterValue("0 < diag(Sigma)")(res, all_pos_definite)
    return res


@_logprob.register(arb.DirichletRV)
def dirichlet_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (alpha,) = inputs[3:]
    res = at.sum(xlogy0(alpha - 1, value) - at.gammaln(alpha), axis=-1) + at.gammaln(
        at.sum(alpha, axis=-1)
    )
    res = at.switch(
        at.bitwise_and(
            at.all(at.le(0.0, value), axis=-1), at.all(at.le(value, 1.0), axis=-1)
        ),
        res,
        -np.inf,
    )
    res = CheckParameterValue("0 < alpha")(res, at.all(at.lt(0.0, alpha)))
    return res


@_logprob.register(arb.MultinomialRV)
def multinomial_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    n, p = inputs[3:]
    res = at.gammaln(n + 1) + at.sum(-at.gammaln(value + 1) + xlogy0(value, p), axis=-1)
    res = at.switch(
        at.bitwise_and(
            at.all(at.le(0.0, value), axis=-1), at.eq(at.sum(value, axis=-1), n)
        ),
        res,
        -np.inf,
    )
    res = CheckParameterValue("p <= 1, sum(p) == 1, n >= 0")(
        res,
        at.all(at.le(p, 1)),
        at.all(at.eq(at.sum(p, axis=-1), 1)),
        at.all(at.ge(n, 0)),
    )
    return res


@_logprob.register(DiracDelta)
def diracdelta_logprob(op, values, *inputs, **kwargs):
    (values,) = values
    (const_value,) = inputs
    values, const_value = at.broadcast_arrays(values, const_value)
    return at.switch(
        at.isclose(values, const_value, rtol=op.rtol, atol=op.atol), 0.0, -np.inf
    )


@_logprob.register(DiscreteMarkovChainFactory)
def discrete_mc_logp(op, values, *inputs, **kwargs):
    r"""Create a Aesara graph that computes the log-likelihood for a discrete Markov chain.

    This is the log-likelihood for the joint distribution of states, :math:`S_t`, conditional
    on state samples, :math:`s_t`, given by the following:

    .. math::

        \int_{S_0} P(S_1 = s_1 \mid S_0) dP(S_0) \prod^{T}_{t=2} P(S_t = s_t \mid S_{t-1} = s_{t-1})

    The first term (i.e. the integral) simply computes the marginal :math:`P(S_1 = s_1)`, so
    another way to express this result is as follows:

    .. math::

        P(S_1 = s_1) \prod^{T}_{t=2} P(S_t = s_t \mid S_{t-1} = s_{t-1})

    XXX TODO: This does not implement complete broadcasting support!

    """

    (states,) = values
    _, Gammas, gamma_0 = inputs[: len(inputs) - len(op.shared_inputs)]

    if states.ndim != 1 or Gammas.ndim > 3 or gamma_0.ndim > 1:
        raise NotImplementedError()

    Gammas_at = at.broadcast_to(Gammas, (states.shape[0],) + tuple(Gammas.shape)[-2:])
    gamma_0_at = gamma_0

    Gamma_1_at = Gammas_at[0]
    P_S_1_at = at.dot(gamma_0_at, Gamma_1_at)[states[0]]

    # def S_logp_fn(S_tm1, S_t, Gamma):
    #     return at.log(Gamma[..., S_tm1, S_t])
    #
    # P_S_2T_at, _ = theano.scan(
    #     S_logp_fn,
    #     sequences=[
    #         {
    #             "input": states_at,
    #             "taps": [-1, 0],
    #         },
    #         Gammas_at,
    #     ],
    # )
    P_S_2T_at = Gammas_at[at.arange(0, states.shape[0] - 1), states[:-1], states[1:]]

    log_P_S_1T_at = at.concatenate(
        [at.shape_padright(at.log(P_S_1_at)), at.log(P_S_2T_at)]
    )

    return log_P_S_1T_at
