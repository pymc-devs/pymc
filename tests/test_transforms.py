import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy as sp
import scipy.special
from aesara.graph.basic import equal_computations
from aesara.graph.fg import FunctionGraph
from numdifftools import Jacobian

from aeppl.joint_logprob import factorized_joint_logprob, joint_logprob
from aeppl.transforms import (
    DEFAULT_TRANSFORM,
    ChainedTransform,
    ExpTransform,
    IntervalTransform,
    LocTransform,
    LogOddsTransform,
    LogTransform,
    RVTransform,
    ScaleTransform,
    TransformValuesMapping,
    TransformValuesOpt,
    _default_transformed_rv,
    transformed_variable,
)
from tests.utils import assert_no_rvs


class DirichletScipyDist:
    def __init__(self, alphas):
        self.alphas = alphas

    def rvs(self, size=None, random_state=None):
        if size is None:
            size = ()
        samples_shape = tuple(np.atleast_1d(size)) + self.alphas.shape
        samples = np.empty(samples_shape)
        alphas_bcast = np.broadcast_to(self.alphas, samples_shape)

        for index in np.ndindex(*samples_shape[:-1]):
            samples[index] = random_state.dirichlet(alphas_bcast[index])

        return samples

    def logpdf(self, value):
        res = np.sum(
            scipy.special.xlogy(self.alphas - 1, value)
            - scipy.special.gammaln(self.alphas),
            axis=-1,
        ) + scipy.special.gammaln(np.sum(self.alphas, axis=-1))
        return res


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
            (np.array([0.7, 0.3]),),
            lambda alpha: sp.stats.dirichlet(alpha),
            (),
        ),
        (
            at.random.dirichlet,
            (np.array([[0.7, 0.3], [0.9, 0.1]]),),
            lambda alpha: DirichletScipyDist(alpha),
            (),
        ),
        pytest.param(
            at.random.dirichlet,
            (np.array([0.3, 0.7]),),
            lambda alpha: DirichletScipyDist(alpha),
            (3, 2),
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
    res = joint_logprob({a: a_value_var, b: b_value_var}, extra_rewrites=transform_opt)

    test_val_rng = np.random.RandomState(3238)

    logp_vals_fn = aesara.function([a_value_var, b_value_var], res)

    a_trans_op = _default_transformed_rv(a.owner.op, a.owner).op
    transform = a_trans_op.transform

    a_forward_fn = aesara.function(
        [a_value_var], transform.forward(a_value_var, *a.owner.inputs)
    )
    a_backward_fn = aesara.function(
        [a_value_var], transform.backward(a_value_var, *a.owner.inputs)
    )
    log_jac_fn = aesara.function(
        [a_value_var],
        transform.log_jac_det(a_value_var, *a.owner.inputs),
        on_unused_input="ignore",
    )

    for i in range(10):
        a_dist = sp_dist(*dist_params)
        a_val = a_dist.rvs(size=size, random_state=test_val_rng).astype(
            a_value_var.dtype
        )
        b_dist = sp.stats.norm(a_val, 1.0)
        b_val = b_dist.rvs(random_state=test_val_rng).astype(b_value_var.dtype)

        a_trans_value = a_forward_fn(a_val)

        if a_val.ndim > 0:

            def jacobian_estimate_novec(value):

                dim_diff = a_val.ndim - value.ndim
                if dim_diff > 0:
                    # Make sure the dimensions match the expected input
                    # dimensions for the compiled backward transform function
                    def a_backward_fn_(x):
                        x_ = np.expand_dims(x, axis=list(range(dim_diff)))
                        return a_backward_fn(x_).squeeze()

                else:
                    a_backward_fn_ = a_backward_fn

                jacobian_val = Jacobian(a_backward_fn_)(value)

                n_missing_dims = jacobian_val.shape[0] - jacobian_val.shape[1]
                if n_missing_dims > 0:
                    missing_bases = np.eye(jacobian_val.shape[0])[..., -n_missing_dims:]
                    jacobian_val = np.concatenate(
                        [jacobian_val, missing_bases], axis=-1
                    )

                return np.linalg.slogdet(jacobian_val)[-1]

            jacobian_estimate = np.vectorize(
                jacobian_estimate_novec, signature="(n)->()"
            )

            exp_log_jac_val = jacobian_estimate(a_trans_value)
        else:
            jacobian_val = np.atleast_2d(
                sp.misc.derivative(a_backward_fn, a_trans_value, dx=1e-6)
            )
            exp_log_jac_val = np.linalg.slogdet(jacobian_val)[-1]

        log_jac_val = log_jac_fn(a_trans_value)
        np.testing.assert_almost_equal(exp_log_jac_val, log_jac_val, decimal=4)

        exp_logprob_val = a_dist.logpdf(a_val).sum()
        exp_logprob_val += exp_log_jac_val.sum()
        exp_logprob_val += b_dist.logpdf(b_val).sum()

        logprob_val = logp_vals_fn(a_trans_value, b_val)

        np.testing.assert_almost_equal(exp_logprob_val, logprob_val, decimal=4)


@pytest.mark.parametrize("use_jacobian", [True, False])
def test_simple_transformed_logprob_nojac(use_jacobian):
    X_rv = at.random.halfnormal(0, 3, name="X")
    x_vv = X_rv.clone()
    x_vv.name = "x"

    transform_opt = TransformValuesOpt({x_vv: DEFAULT_TRANSFORM})
    tr_logp = joint_logprob(
        {X_rv: x_vv}, extra_rewrites=transform_opt, use_jacobian=use_jacobian
    )

    assert np.isclose(
        tr_logp.eval({x_vv: np.log(2.5)}),
        sp.stats.halfnorm(0, 3).logpdf(2.5) + (np.log(2.5) if use_jacobian else 0.0),
    )


@pytest.mark.parametrize("ndim", (0, 1))
def test_fallback_log_jac_det(ndim):
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

    value = at.TensorType("float64", (None,) * ndim)("value")
    value_tr = square_tr.forward(value)
    log_jac_det = square_tr.log_jac_det(value_tr)

    test_value = np.full((2,) * ndim, 3)
    expected_log_jac_det = -np.log(6) * test_value.size
    assert np.isclose(log_jac_det.eval({value: test_value}), expected_log_jac_det)


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


def test_original_values_output_dict():
    """
    Test that the original unconstrained value variable appears an the key of
    the logprob factor
    """
    p_rv = at.random.beta(1, 1, name="p")
    p_vv = p_rv.clone()

    tr = TransformValuesOpt({p_vv: DEFAULT_TRANSFORM})
    logp_dict = factorized_joint_logprob({p_rv: p_vv}, extra_rewrites=tr)

    assert p_vv in logp_dict


def test_mixture_transform():
    """Make sure that non-`RandomVariable` `MeasurableVariable`s can be transformed.

    This test is specific to `MixtureRV`, which is derived from an `OpFromGraph`.
    """

    I_rv = at.random.bernoulli(0.5, name="I")
    Y_1_rv = at.random.beta(100, 1, name="Y_1")
    Y_2_rv = at.random.beta(1, 100, name="Y_2")

    # A `MixtureRV`, which is an `OpFromGraph` subclass, will replace this
    # `at.stack` in the graph
    Y_rv = at.stack([Y_1_rv, Y_2_rv])[I_rv]
    Y_rv.name = "Y"

    i_vv = I_rv.clone()
    i_vv.name = "i"
    y_vv = Y_rv.clone()
    y_vv.name = "y"

    logp_no_trans = joint_logprob(
        {Y_rv: y_vv, I_rv: i_vv},
    )

    transform_opt = TransformValuesOpt({y_vv: LogTransform()})

    with pytest.warns(None) as record:
        # This shouldn't raise any warnings
        logp_trans = joint_logprob(
            {Y_rv: y_vv, I_rv: i_vv},
            extra_rewrites=transform_opt,
            use_jacobian=False,
        )

    assert not record.list

    # The untransformed graph should be the same as the transformed graph after
    # replacing the `Y_rv` value variable with a transformed version of itself
    logp_nt_fg = FunctionGraph(outputs=[logp_no_trans], clone=False)
    y_trans = transformed_variable(at.exp(y_vv), y_vv)
    y_trans.name = "y_log"
    logp_nt_fg.replace(y_vv, y_trans)
    logp_nt = logp_nt_fg.outputs[0]

    assert equal_computations([logp_nt], [logp_trans])


def test_invalid_interval_transform():
    x_rv = at.random.normal(0, 1)
    x_vv = x_rv.clone()

    msg = "Both edges of IntervalTransform cannot be None"
    tr = IntervalTransform(lambda *inputs: (None, None))
    with pytest.raises(ValueError, match=msg):
        tr.forward(x_vv, *x_rv.owner.inputs)

    tr = IntervalTransform(lambda *inputs: (None, None))
    with pytest.raises(ValueError, match=msg):
        tr.backward(x_vv, *x_rv.owner.inputs)

    tr = IntervalTransform(lambda *inputs: (None, None))
    with pytest.raises(ValueError, match=msg):
        tr.log_jac_det(x_vv, *x_rv.owner.inputs)


def test_chained_transform():
    loc = 5
    scale = 0.1

    ch = ChainedTransform(
        transform_list=[
            ScaleTransform(
                transform_args_fn=lambda *inputs: at.constant(scale),
            ),
            ExpTransform(),
            LocTransform(
                transform_args_fn=lambda *inputs: at.constant(loc),
            ),
        ],
        base_op=at.random.multivariate_normal,
    )

    x = at.random.multivariate_normal(np.zeros(3), np.eye(3))
    x_val = x.eval()

    x_val_forward = ch.forward(x_val, *x.owner.inputs).eval()
    assert np.allclose(
        x_val_forward,
        np.exp(x_val * scale) + loc,
    )

    x_val_backward = ch.backward(x_val_forward, *x.owner.inputs, scale, loc).eval()
    assert np.allclose(
        x_val_backward,
        x_val,
    )

    log_jac_det = ch.log_jac_det(x_val_forward, *x.owner.inputs, scale, loc)
    assert np.isclose(
        log_jac_det.eval(),
        -np.log(scale) - np.sum(np.log(x_val_forward - loc)),
    )
