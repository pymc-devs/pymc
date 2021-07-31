import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy as sp
from aesara.graph.fg import FunctionGraph
from numdifftools import Jacobian

from aeppl.joint_logprob import joint_logprob
from aeppl.transforms import (
    DEFAULT_TRANSFORM,
    LogOddsTransform,
    LogTransform,
    RVTransform,
    TransformValuesMapping,
    TransformValuesOpt,
    _default_transformed_rv,
)
from tests.utils import assert_no_rvs


@pytest.mark.parametrize(
    "at_dist, dist_params, sp_dist, size",
    [
        (at.random.uniform, (0, 1), sp.stats.uniform, ()),
        (
            at.random.pareto,
            (1.5, 10.5),
            lambda b, scale: sp.stats.pareto(b, scale=scale),
            (),
        ),
        (
            at.random.triangular,
            (1.5, 3.0, 10.5),
            lambda lower, mode, upper: sp.stats.triang(
                (mode - lower) / (upper - lower), loc=lower, scale=upper - lower
            ),
            (),
        ),
        (
            at.random.halfnormal,
            (0, 1),
            sp.stats.halfnorm,
            (),
        ),
        (
            at.random.wald,
            (1.5, 10.5),
            lambda mean, scale: sp.stats.invgauss(mean / scale, scale=scale),
            (),
        ),
        (
            at.random.exponential,
            (1.5,),
            lambda mu: sp.stats.expon(scale=mu),
            (),
        ),
        pytest.param(
            at.random.lognormal,
            (-1.5, 10.5),
            lambda mu, sigma: sp.stats.lognorm(s=sigma, loc=0, scale=np.exp(mu)),
            (),
        ),
        (
            at.random.lognormal,
            (-1.5, 1.5),
            lambda mu, sigma: sp.stats.lognorm(s=sigma, scale=np.exp(mu)),
            (),
        ),
        (
            at.random.halfcauchy,
            (1.5, 10.5),
            lambda alpha, beta: sp.stats.halfcauchy(loc=alpha, scale=beta),
            (),
        ),
        (
            at.random.gamma,
            (1.5, 10.5),
            lambda alpha, inv_beta: sp.stats.gamma(alpha, scale=1.0 / inv_beta),
            (),
        ),
        (
            at.random.invgamma,
            (1.5, 10.5),
            lambda alpha, beta: sp.stats.invgamma(alpha, scale=beta),
            (),
        ),
        (
            at.random.chisquare,
            (1.5,),
            lambda df: sp.stats.chi2(df),
            (),
        ),
        (
            at.random.weibull,
            (1.5, 10.5),
            lambda alpha, beta: sp.stats.weibull_min(alpha, scale=beta),
            (),
        ),
        (
            at.random.beta,
            (1.5, 1.5),
            lambda alpha, beta: sp.stats.beta(alpha, beta),
            (),
        ),
        (
            at.random.vonmises,
            (1.5, 10.5),
            lambda mu, kappa: sp.stats.vonmises(kappa, loc=mu),
            (),
        ),
        (
            at.random.dirichlet,
            (np.array([0.5, 0.5]),),
            lambda alpha: sp.stats.dirichlet(alpha),
            (),
        ),
        pytest.param(
            at.random.dirichlet,
            (np.array([0.5, 0.5]),),
            lambda alpha: sp.stats.dirichlet(alpha),
            (3, 2),
            marks=pytest.mark.xfail(
                reason="Need to make the test framework work for arbitrary sizes"
            ),
        ),
    ],
)
def test_transformed_logprob(at_dist, dist_params, sp_dist, size):
    """
    This test takes a `RandomVariable` type, plus parameters, and uses it to
    construct a variable ``a`` that's used in the graph ``b =
    at.random.normal(a, 1.0)``.  The transformed log-probability is then
    computed for ``b``.  We then test that the log-probability of ``a`` is
    properly transformed, as well as any instances of ``a`` that are used
    elsewhere in the graph (i.e. in ``b``), by comparing the graph for the
    transformed log-probability with the SciPy-derived log-probability--using a
    numeric approximation to the Jacobian term.
    """

    a = at_dist(*dist_params, size=size)
    a.name = "a"
    a_value_var = a.clone()
    a_value_var.name = "a_value"

    b = at.random.normal(a, 1.0)
    b.name = "b"
    b_value_var = b.clone()
    b_value_var.name = "b_value"

    transform_opt = TransformValuesOpt({a_value_var: DEFAULT_TRANSFORM})
    res = joint_logprob(
        b, {a: a_value_var, b: b_value_var}, extra_rewrites=transform_opt
    )

    test_val_rng = np.random.RandomState(3238)

    decimals = 6 if aesara.config.floatX == "float64" else 4
    logp_vals_fn = aesara.function([a_value_var, b_value_var], res)

    a_trans_op = _default_transformed_rv(a.owner.op, a.owner).op
    transform = a_trans_op.transform

    a_forward_fn = aesara.function(
        [a_value_var], transform.forward(a_value_var, *a.owner.inputs)
    )
    a_backward_fn = aesara.function(
        [a_value_var], transform.backward(a_value_var, *a.owner.inputs)
    )

    for i in range(10):
        a_dist = sp_dist(*dist_params)
        a_val = a_dist.rvs(size=size, random_state=test_val_rng).astype(
            a_value_var.dtype
        )
        b_dist = sp.stats.norm(a_val, 1.0)
        b_val = b_dist.rvs(random_state=test_val_rng).astype(b_value_var.dtype)

        exp_logprob_val = a_dist.logpdf(a_val)

        a_trans_value = a_forward_fn(a_val)
        if a_val.ndim > 0:
            # exp_logprob_val = np.vectorize(a_dist.logpdf, signature="(n)->()")(a_val)
            jacobian_val = Jacobian(a_backward_fn)(a_trans_value)[:-1]
        else:
            jacobian_val = np.atleast_2d(
                sp.misc.derivative(a_backward_fn, a_trans_value, dx=1e-6)
            )

        exp_logprob_val += np.log(np.linalg.det(jacobian_val))
        exp_logprob_val += b_dist.logpdf(b_val)

        logprob_val = logp_vals_fn(a_trans_value, b_val)

        np.testing.assert_almost_equal(exp_logprob_val, logprob_val, decimal=decimals)


def test_simple_transformed_logprob():
    x_rv = at.random.halfnormal(0, 3, name="x_rv")
    x = x_rv.clone()

    transform_opt = TransformValuesOpt({x: DEFAULT_TRANSFORM})
    tr_logp = joint_logprob(x_rv, {x_rv: x}, extra_rewrites=transform_opt)

    assert np.isclose(
        tr_logp.eval({x: np.log(2.5)}),
        sp.stats.halfnorm(0, 3).logpdf(2.5) + np.log(2.5),
    )


def test_fallback_log_jac_det():
    """
    Test fallback log_jac_det in RVTransform produces correct the graph for a
    simple transformation: x**2 -> -log(2*x)
    """

    class SquareTransform(RVTransform):
        name = "square"

        def forward(self, value, *inputs):
            return at.power(value, 2)

        def backward(self, value, *inputs):
            return at.sqrt(value)

    square_tr = SquareTransform()

    value = at.scalar("value")
    value_tr = square_tr.forward(value)
    log_jac_det = square_tr.log_jac_det(value_tr)

    assert np.isclose(log_jac_det.eval({value: 3}), -np.log(6))


def test_hierarchical_uniform_transform():
    """
    This model requires rv-value replacements in the backward transformation of
    the value var `x`
    """

    lower_rv = at.random.uniform(0, 1, name="lower")
    upper_rv = at.random.uniform(9, 10, name="upper")
    x_rv = at.random.uniform(lower_rv, upper_rv, name="x")

    lower = lower_rv.clone()
    upper = upper_rv.clone()
    x = x_rv.clone()

    transform_opt = TransformValuesOpt(
        {
            lower: DEFAULT_TRANSFORM,
            upper: DEFAULT_TRANSFORM,
            x: DEFAULT_TRANSFORM,
        }
    )
    logp = joint_logprob(
        x_rv,
        {lower_rv: lower, upper_rv: upper, x_rv: x},
        extra_rewrites=transform_opt,
    )

    assert_no_rvs(logp)
    assert not np.isinf(logp.eval({lower: -10, upper: 20, x: -20}))


def test_nondefault_transforms():
    loc_rv = at.random.uniform(-10, 10, name="loc")
    scale_rv = at.random.uniform(-1, 1, name="scale")
    x_rv = at.random.normal(loc_rv, scale_rv, name="x")

    loc = loc_rv.clone()
    scale = scale_rv.clone()
    x = x_rv.clone()

    transform_opt = TransformValuesOpt(
        {
            loc: None,
            scale: LogOddsTransform(),
            x: LogTransform(),
        }
    )

    logp = joint_logprob(
        x_rv,
        {loc_rv: loc, scale_rv: scale, x_rv: x},
        extra_rewrites=transform_opt,
    )

    # Check numerical evaluation matches with expected transforms
    loc_val = 0
    scale_val_tr = -1
    x_val_tr = -1

    scale_val = sp.special.expit(scale_val_tr)
    x_val = np.exp(x_val_tr)

    exp_logp = 0
    exp_logp += sp.stats.uniform(-10, 20).logpdf(loc_val)
    exp_logp += sp.stats.uniform(-1, 2).logpdf(scale_val)
    exp_logp += np.log(scale_val) + np.log1p(-scale_val)  # logodds log_jac_det
    exp_logp += sp.stats.norm(loc_val, scale_val).logpdf(x_val)
    exp_logp += x_val_tr  # log log_jac_det

    assert np.isclose(
        logp.eval({loc: loc_val, scale: scale_val_tr, x: x_val_tr}),
        exp_logp,
    )


def test_default_transform_multiout():
    r"""Make sure that `Op`\s with multiple outputs are handled correctly."""

    # This SVD value is necessarily `1`, but it's generated by an `Op` with
    # multiple outputs and no default output.
    sd = at.linalg.svd(at.eye(1))[1][0]
    x_rv = at.random.normal(0, sd, name="x")
    x = x_rv.clone()

    transform_opt = TransformValuesOpt({x: DEFAULT_TRANSFORM})

    logp = joint_logprob(
        x_rv,
        {x_rv: x},
        extra_rewrites=transform_opt,
    )

    assert np.isclose(
        logp.eval({x: 1}),
        sp.stats.norm(0, 1).logpdf(1),
    )


def test_nonexistent_default_transform():
    """
    Test that setting `DEFAULT_TRANSFORM` to a variable that has no default
    transform does not fail
    """
    x_rv = at.random.normal(name="x")
    x = x_rv.clone()

    transform_opt = TransformValuesOpt({x: DEFAULT_TRANSFORM})

    logp = joint_logprob(
        x_rv,
        {x_rv: x},
        extra_rewrites=transform_opt,
    )

    assert np.isclose(
        logp.eval({x: 1}),
        sp.stats.norm(0, 1).logpdf(1),
    )


def test_TransformValuesMapping():
    x = at.vector()
    fg = FunctionGraph(outputs=[x])

    tvm = TransformValuesMapping({})
    fg.attach_feature(tvm)

    tvm2 = TransformValuesMapping({})
    fg.attach_feature(tvm2)

    assert fg._features[-1] is tvm
