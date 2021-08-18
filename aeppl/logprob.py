from functools import singledispatch
from typing import Tuple

import aesara.tensor as at
import aesara.tensor.random.basic as arb
import numpy as np
from aesara.assert_op import Assert
from aesara.graph.op import Op
from aesara.tensor.slinalg import Cholesky, solve_lower_triangular
from aesara.tensor.var import TensorVariable

# from aesara.tensor.xlogx import xlogy0

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
    raise NotImplementedError()


@_logprob.register(arb.UniformRV)
def uniform_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    lower, upper = inputs[3:]
    return at.switch(
        at.bitwise_and(at.ge(value, lower), at.le(value, upper)),
        at.fill(value, -at.log(upper - lower)),
        -np.inf,
    )


@_logprob.register(arb.NormalRV)
def normal_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, sigma = inputs[3:]
    res = (
        -0.5 * at.pow((value - mu) / sigma, 2)
        - at.log(np.sqrt(2.0 * np.pi))
        - at.log(sigma)
    )
    res = Assert("sigma > 0")(res, at.all(at.gt(sigma, 0.0)))
    return res


@_logprob.register(arb.HalfNormalRV)
def halfnormal_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    loc, sigma = inputs[3:]
    res = (
        -0.5 * at.pow((value - loc) / sigma, 2)
        + at.log(np.sqrt(2.0 / np.pi))
        - at.log(sigma)
    )
    res = at.switch(at.ge(value, loc), res, -np.inf)
    res = Assert("sigma > 0")(res, at.all(at.gt(sigma, 0.0)))
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
    res = Assert("0 <= value <= 1, alpha > 0, beta > 0")(
        res, at.all(at.gt(alpha, 0.0)), at.all(at.gt(beta, 0.0))
    )
    return res


@_logprob.register(arb.ExponentialRV)
def exponential_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (mu,) = inputs[3:]
    res = -at.log(mu) - value / mu
    res = at.switch(at.ge(value, 0.0), res, -np.inf)
    res = Assert("mu > 0")(res, at.all(at.gt(mu, 0.0)))
    return res


@_logprob.register(arb.LaplaceRV)
def laplace_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, b = inputs[3:]
    return -at.log(2 * b) - at.abs_(value - mu) / b


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
    res = Assert("sigma > 0")(res, at.all(at.gt(sigma, 0)))
    return res


@_logprob.register(arb.ParetoRV)
def pareto_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    alpha, m = inputs[3:]
    res = at.log(alpha) + xlogy0(alpha, m) - xlogy0(alpha + 1.0, value)
    res = at.switch(at.ge(value, m), res, -np.inf)
    res = Assert("alpha > 0, m > 0")(
        res, at.all(at.gt(alpha, 0.0)), at.all(at.gt(m, 0.0))
    )
    return res


@_logprob.register(arb.CauchyRV)
def cauchy_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    alpha, beta = inputs[3:]
    res = -at.log(np.pi) - at.log(beta) - at.log1p(at.pow((value - alpha) / beta, 2))
    res = Assert("beta > 0")(res, at.all(at.gt(beta, 0.0)))
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
    res = Assert("alpha > 0, beta > 0")(
        res, at.all(at.gt(alpha, 0.0)), at.all(at.gt(beta, 0.0))
    )
    return res


@_logprob.register(arb.InvGammaRV)
def invgamma_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    alpha, beta = inputs[3:]
    res = -(alpha + 1) * np.log(value) - at.gammaln(alpha) - 1.0 / value
    res = (
        -at.gammaln(alpha)
        + xlogy0(alpha, beta)
        - beta / value
        + xlogy0(-alpha - 1, value)
    )
    res = at.switch(at.ge(value, 0.0), res, -np.inf)
    res = Assert("alpha > 0, beta > 0")(
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
    res = Assert("mu > 0, scale > 0")(
        res, at.all(at.gt(mu, 0.0)), at.all(at.gt(scale, 0.0))
    )
    return res


@_logprob.register(arb.WeibullRV)
def weibull_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    alpha, beta = inputs[3:]
    res = (
        at.log(alpha)
        - at.log(beta)
        + (alpha - 1.0) * at.log(value / beta)
        - at.pow(value / beta, alpha)
    )
    res = at.switch(at.ge(value, 0.0), res, -np.inf)
    res = Assert("alpha > 0, beta > 0")(
        res, at.all(at.gt(alpha, 0.0)), at.all(at.gt(beta, 0.0))
    )
    return res


@_logprob.register(arb.VonMisesRV)
def vonmises_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, kappa = inputs[3:]
    res = kappa * at.cos(mu - value) - at.log(2 * np.pi) - at.log(at.i0(kappa))
    res = at.switch(
        at.bitwise_and(at.ge(value, -np.pi), at.le(value, np.pi)), res, -np.inf
    )
    res = Assert("kappa > 0")(res, at.all(at.gt(kappa, 0.0)))
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
    res = Assert("lower <= c, c <= upper")(
        res, at.all(at.le(lower, c)), at.all(at.le(c, upper))
    )
    return res


@_logprob.register(arb.GumbelRV)
def gumbel_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, beta = inputs[3:]
    z = (value - mu) / beta
    res = -z - at.exp(-z) - at.log(beta)
    res = Assert("0 < beta")(res, at.all(at.lt(0.0, beta)))
    return res


@_logprob.register(arb.LogisticRV)
def logistic_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    mu, s = inputs[3:]
    z = (value - mu) / s
    res = -z - at.log(s) - 2.0 * at.log1p(at.exp(-z))
    res = Assert("0 < s")(res, at.all(at.lt(0.0, s)))
    return res


@_logprob.register(arb.BinomialRV)
def binomial_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    n, p = inputs[3:]
    res = binomln(n, value) + xlogy0(value, p) + xlogy0(n - value, 1.0 - p)
    res = at.switch(at.bitwise_and(at.le(0, value), at.le(value, n)), res, -np.inf)
    res = Assert("0 <= p, p <= 1")(res, at.all(at.le(0.0, p)), at.all(at.le(p, 1.0)))
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
    res = Assert("0 < alpha, 0 < beta")(
        res, at.all(at.lt(0.0, alpha)), at.all(at.lt(0.0, beta))
    )
    return res


@_logprob.register(arb.BernoulliRV)
def bernoulli_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (p,) = inputs[3:]
    res = at.switch(value, at.log(p), at.log(1.0 - p))
    res = at.switch(at.bitwise_and(at.le(0, value), at.le(value, 1)), res, -np.inf)
    res = Assert("0 <= p <= 1")(res, at.all(at.le(0.0, p)), at.all(at.le(p, 1.0)))
    return res


@_logprob.register(arb.PoissonRV)
def poisson_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (mu,) = inputs[3:]
    res = xlogy0(value, mu) - at.gammaln(value + 1) - mu
    res = at.switch(at.le(0, value), res, -np.inf)
    res = Assert("0 <= mu")(res, at.all(at.le(0.0, mu)))
    res = at.switch(at.bitwise_and(at.eq(mu, 0.0), at.eq(value, 0.0)), 0.0, res)
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
    res = Assert("0 < mu, 0 < n")(res, at.all(at.lt(0.0, mu)), at.all(at.lt(0.0, n)))
    res = at.switch(at.gt(n, 1e10), poisson_logprob(op, values, *inputs[:3], mu), res)
    return res


@_logprob.register(arb.GeometricRV)
def geometric_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (p,) = inputs[3:]
    res = at.log(p) + xlogy0(value - 1, 1 - p)
    res = at.switch(at.le(1, value), res, -np.inf)
    res = Assert("0 <= p <= 1")(res, at.all(at.le(0.0, p)), at.all(at.ge(1.0, p)))
    return res


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
    res = Assert("0 <= p <= 1")(res, at.all(at.ge(p, 0.0)), at.all(at.le(p, 1.0)))
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
    res = -0.5 * n * np.log(2 * np.pi) - 0.5 * quaddist - logdet
    res = Assert("0 < diag(Sigma)")(res, all_pos_definite)
    return res


@_logprob.register(arb.DirichletRV)
def dirichlet_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    (alpha,) = inputs[3:]
    res = at.sum(at.gammaln(alpha)) - at.gammaln(at.sum(alpha))
    res = -res + at.sum((xlogy0(alpha - 1, value.T)).T, axis=0)
    res = at.switch(
        at.bitwise_and(
            at.all(at.le(0.0, value), axis=-1), at.all(at.le(value, 1.0), axis=-1)
        ),
        res,
        -np.inf,
    )
    res = Assert("0 < alpha")(res, at.all(at.lt(0.0, alpha)))
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
    res = Assert("p <= 1, sum(p) == 1, n >= 0")(
        res,
        at.all(at.le(p, 1)),
        at.all(at.eq(at.sum(p, axis=-1), 1)),
        at.all(at.ge(n, 0)),
    )
    return res
