#   Copyright 2024 The PyMC Developers
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
import gc
import operator

import numpy as np
import pytensor
import pytest
import scipy as sp

from numdifftools import Jacobian
from pytensor import scan
from pytensor import tensor as pt
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import FunctionGraph
from pytensor.graph.basic import equal_computations

import pymc as pm

from pymc.distributions.transforms import _default_transform, log, logodds
from pymc.logprob import conditional_logp
from pymc.logprob.abstract import MeasurableVariable, _logprob
from pymc.logprob.transform_value import TransformValuesMapping, TransformValuesRewrite
from pymc.logprob.transforms import ExpTransform, LogOddsTransform, LogTransform
from pymc.testing import assert_no_rvs
from tests.logprob.test_transforms import DirichletScipyDist


@pytest.fixture(scope="module")
def multiout_measurable_op():
    # Create a dummy Op that just returns the two inputs
    mu1, mu2 = pt.scalars("mu1", "mu2")

    class TestOpFromGraph(OpFromGraph):
        def do_constant_folding(self, fgraph, node):
            False

    multiout_op = TestOpFromGraph([mu1, mu2], [mu1 + 0.0, mu2 + 0.0])

    MeasurableVariable.register(TestOpFromGraph)

    @_logprob.register(TestOpFromGraph)
    def logp_multiout(op, values, mu1, mu2):
        value1, value2 = values
        return value1 + mu1, value2 + mu2

    return multiout_op


def test_TransformValuesMapping():
    x = pt.vector()
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
    p_rv = pt.random.beta(1, 1, name="p")
    p_vv = p_rv.clone()

    tr = TransformValuesRewrite({p_vv: logodds})
    logp_dict = conditional_logp({p_rv: p_vv}, extra_rewrites=tr)

    assert p_vv in logp_dict


@pytest.mark.parametrize(
    "pt_dist, dist_params, sp_dist, size",
    [
        (pt.random.uniform, (0, 1), sp.stats.uniform, ()),
        (
            pt.random.pareto,
            (1.5, 10.5),
            lambda b, scale: sp.stats.pareto(b, scale=scale),
            (),
        ),
        (
            pt.random.triangular,
            (1.5, 3.0, 10.5),
            lambda lower, mode, upper: sp.stats.triang(
                (mode - lower) / (upper - lower), loc=lower, scale=upper - lower
            ),
            (),
        ),
        (
            pt.random.halfnormal,
            (0, 1),
            sp.stats.halfnorm,
            (),
        ),
        pytest.param(
            pt.random.wald,
            (1.5, 10.5),
            lambda mean, scale: sp.stats.invgauss(mean / scale, scale=scale),
            (),
            marks=pytest.mark.xfail(
                reason="We don't use PyTensor's Wald operator",
                raises=NotImplementedError,
            ),
        ),
        (
            pt.random.exponential,
            (1.5,),
            lambda mu: sp.stats.expon(scale=mu),
            (),
        ),
        pytest.param(
            pt.random.lognormal,
            (-1.5, 10.5),
            lambda mu, sigma: sp.stats.lognorm(s=sigma, loc=0, scale=np.exp(mu)),
            (),
        ),
        (
            pt.random.lognormal,
            (-1.5, 1.5),
            lambda mu, sigma: sp.stats.lognorm(s=sigma, scale=np.exp(mu)),
            (),
        ),
        (
            pt.random.halfcauchy,
            (1.5, 10.5),
            lambda alpha, beta: sp.stats.halfcauchy(loc=alpha, scale=beta),
            (),
        ),
        (
            pt.random.gamma,
            (1.5, 10.5),
            lambda alpha, inv_beta: sp.stats.gamma(alpha, scale=1.0 / inv_beta),
            (),
        ),
        (
            pt.random.invgamma,
            (1.5, 10.5),
            lambda alpha, beta: sp.stats.invgamma(alpha, scale=beta),
            (),
        ),
        (
            pm.ChiSquared.dist,
            (1.5,),
            lambda df: sp.stats.chi2(df),
            (),
        ),
        pytest.param(
            pt.random.weibull,
            (1.5,),
            lambda c: sp.stats.weibull_min(c),
            (),
            marks=pytest.mark.xfail(
                reason="We don't use PyTensor's Weibull operator",
                raises=NotImplementedError,
            ),
        ),
        (
            pt.random.beta,
            (1.5, 1.5),
            lambda alpha, beta: sp.stats.beta(alpha, beta),
            (),
        ),
        (
            pt.random.vonmises,
            (1.5, 10.5),
            lambda mu, kappa: sp.stats.vonmises(kappa, loc=mu),
            (),
        ),
        (
            pt.random.dirichlet,
            (np.array([0.7, 0.3]),),
            lambda alpha: sp.stats.dirichlet(alpha),
            (),
        ),
        (
            pt.random.dirichlet,
            (np.array([[0.7, 0.3], [0.9, 0.1]]),),
            lambda alpha: DirichletScipyDist(alpha),
            (),
        ),
        pytest.param(
            pt.random.dirichlet,
            (np.array([0.3, 0.7]),),
            lambda alpha: DirichletScipyDist(alpha),
            (3, 2),
        ),
    ],
)
def test_default_value_transform_logprob(pt_dist, dist_params, sp_dist, size):
    """
    This test takes a `RandomVariable` type, plus parameters, and uses it to
    construct a variable ``a`` that's used in the graph ``b =
    pt.random.normal(a, 1.0)``.  The transformed log-probability is then
    computed for ``b``.  We then test that the log-probability of ``a`` is
    properly transformed, as well as any instances of ``a`` that are used
    elsewhere in the graph (i.e. in ``b``), by comparing the graph for the
    transformed log-probability with the SciPy-derived log-probability--using a
    numeric approximation to the Jacobian term.

    TODO: This test is rather redundant with those in tess/distributions/test_transform.py
    """

    a = pt_dist(*dist_params, size=size)
    a.name = "a"
    a_value_var = pt.tensor(dtype=a.dtype, shape=(None,) * a.ndim)
    a_value_var.name = "a_value"

    b = pt.random.normal(a, 1.0)
    b.name = "b"
    b_value_var = b.clone()
    b_value_var.name = "b_value"

    transform = _default_transform(a.owner.op, a)
    transform_rewrite = TransformValuesRewrite({a_value_var: transform})
    res = conditional_logp({a: a_value_var, b: b_value_var}, extra_rewrites=transform_rewrite)
    res_combined = pt.sum([pt.sum(factor) for factor in res.values()])

    test_val_rng = np.random.RandomState(3238)

    logp_vals_fn = pytensor.function([a_value_var, b_value_var], res_combined)

    a_forward_fn = pytensor.function([a_value_var], transform.forward(a_value_var, *a.owner.inputs))
    a_backward_fn = pytensor.function(
        [a_value_var], transform.backward(a_value_var, *a.owner.inputs)
    )
    log_jac_fn = pytensor.function(
        [a_value_var],
        transform.log_jac_det(a_value_var, *a.owner.inputs),
        on_unused_input="ignore",
    )

    for i in range(10):
        a_dist = sp_dist(*dist_params)
        a_val = a_dist.rvs(size=size, random_state=test_val_rng).astype(a_value_var.dtype)
        b_dist = sp.stats.norm(a_val, 1.0)
        b_val = b_dist.rvs(random_state=test_val_rng).astype(b_value_var.dtype)

        a_trans_value = a_forward_fn(a_val)

        if a_val.ndim > 0:

            def jacobian_estimate_novec(value):
                dim_diff = a_val.ndim - value.ndim  # pylint: disable=cell-var-from-loop
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
                    jacobian_val = np.concatenate([jacobian_val, missing_bases], axis=-1)

                return np.linalg.slogdet(jacobian_val)[-1]

            jacobian_estimate = np.vectorize(jacobian_estimate_novec, signature="(n)->()")

            exp_log_jac_val = jacobian_estimate(a_trans_value)
        else:
            jacobian_val = np.atleast_2d(sp.misc.derivative(a_backward_fn, a_trans_value, dx=1e-6))
            exp_log_jac_val = np.linalg.slogdet(jacobian_val)[-1]

        log_jac_val = log_jac_fn(a_trans_value)
        np.testing.assert_allclose(exp_log_jac_val, log_jac_val, rtol=1e-4, atol=1e-10)

        exp_logprob_val = a_dist.logpdf(a_val).sum()
        exp_logprob_val += exp_log_jac_val.sum()
        exp_logprob_val += b_dist.logpdf(b_val).sum()

        logprob_val = logp_vals_fn(a_trans_value, b_val)

        np.testing.assert_allclose(exp_logprob_val, logprob_val, rtol=1e-4, atol=1e-10)


@pytest.mark.parametrize("use_jacobian", [True, False])
def test_value_transform_logprob_nojac(use_jacobian):
    X_rv = pt.random.halfnormal(0, 3, name="X")
    x_vv = X_rv.clone()
    x_vv.name = "x"

    transform_rewrite = TransformValuesRewrite({x_vv: log})
    tr_logp = conditional_logp(
        {X_rv: x_vv}, extra_rewrites=transform_rewrite, use_jacobian=use_jacobian
    )
    tr_logp_combined = pt.sum([pt.sum(factor) for factor in tr_logp.values()])

    np.testing.assert_allclose(
        tr_logp_combined.eval({x_vv: np.log(2.5)}),
        sp.stats.halfnorm(0, 3).logpdf(2.5) + (np.log(2.5) if use_jacobian else 0.0),
    )


def test_hierarchical_value_transform():
    """
    This model requires rv-value replacements in the backward transformation of
    the value var `x`
    """

    lower_rv = pt.random.uniform(0, 1, name="lower")
    upper_rv = pt.random.uniform(9, 10, name="upper")
    x_rv = pt.random.uniform(lower_rv, upper_rv, name="x")

    lower = lower_rv.clone()
    upper = upper_rv.clone()
    x = x_rv.clone()

    transform_rewrite = TransformValuesRewrite(
        {
            lower: _default_transform(lower_rv.owner.op, lower_rv),
            upper: _default_transform(upper_rv.owner.op, upper_rv),
            x: _default_transform(x_rv.owner.op, x_rv),
        }
    )
    logp = conditional_logp(
        {lower_rv: lower, upper_rv: upper, x_rv: x},
        extra_rewrites=transform_rewrite,
    )
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

    assert_no_rvs(logp_combined)
    assert not np.isinf(logp_combined.eval({lower: -10, upper: 20, x: -20}))


def test_nondefault_value_transform():
    loc_rv = pt.random.uniform(-10, 10, name="loc")
    scale_rv = pt.random.uniform(-1, 1, name="scale")
    x_rv = pt.random.normal(loc_rv, scale_rv, name="x")

    loc = loc_rv.clone()
    scale = scale_rv.clone()
    x = x_rv.clone()

    transform_rewrite = TransformValuesRewrite(
        {
            loc: None,
            scale: LogOddsTransform(),
            x: LogTransform(),
        }
    )

    logp = conditional_logp(
        {loc_rv: loc, scale_rv: scale, x_rv: x},
        extra_rewrites=transform_rewrite,
    )
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

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

    np.testing.assert_allclose(
        logp_combined.eval({loc: loc_val, scale: scale_val_tr, x: x_val_tr}),
        exp_logp,
    )


def test_no_value_transform_multiout_input():
    r"""Make sure that `Op`\s with multiple outputs are handled correctly."""

    # This SVD value is necessarily `1`, but it's generated by an `Op` with
    # multiple outputs and no default output.
    sd = pt.linalg.svd(pt.eye(1))[1][0]
    x_rv = pt.random.normal(0, sd, name="x")
    x = x_rv.clone()

    transform_rewrite = TransformValuesRewrite({x: None})

    logp = conditional_logp(
        {x_rv: x},
        extra_rewrites=transform_rewrite,
    )
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

    np.testing.assert_allclose(
        logp_combined.eval({x: 1}),
        sp.stats.norm(0, 1).logpdf(1),
    )


@pytest.mark.parametrize("transform_x", (True, False))
@pytest.mark.parametrize("transform_y", (True, False))
def test_value_transform_multiout_op(transform_x, transform_y, multiout_measurable_op):
    x, y = multiout_measurable_op(1, 2)
    x.name = "x"
    y.name = "y"
    x_vv = x.clone()
    y_vv = y.clone()

    transform_rewrite = TransformValuesRewrite(
        {
            x_vv: LogTransform() if transform_x else None,
            y_vv: ExpTransform() if transform_y else None,
        }
    )

    logp = conditional_logp({x: x_vv, y: y_vv}, extra_rewrites=transform_rewrite)
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

    x_vv_test = np.random.normal()
    y_vv_test = np.abs(np.random.normal())

    expected_logp = 0
    if not transform_x:
        expected_logp += x_vv_test + 1
    else:
        expected_logp += np.exp(x_vv_test) + 1 + x_vv_test
    # y logp
    if not transform_y:
        expected_logp += y_vv_test + 2
    else:
        expected_logp += np.log(y_vv_test) + 2 - np.log(y_vv_test)

    np.testing.assert_allclose(
        logp_combined.eval({x_vv: x_vv_test, y_vv: y_vv_test}), expected_logp
    )


def test_transformed_rv_and_value():
    y_rv = pt.random.halfnormal(-1, 1, name="base_rv") + 1
    y_rv.name = "y"
    y_vv = y_rv.clone()

    transform_rewrite = TransformValuesRewrite({y_vv: LogTransform()})

    logp = conditional_logp({y_rv: y_vv}, extra_rewrites=transform_rewrite)
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])
    assert_no_rvs(logp_combined)
    logp_fn = pytensor.function([y_vv], logp_combined)

    y_test_val = -5

    np.testing.assert_allclose(
        logp_fn(y_test_val),
        sp.stats.halfnorm(0, 1).logpdf(np.exp(y_test_val)) + y_test_val,
    )


@pytest.mark.filterwarnings("error")
def test_mixture_transform():
    """Make sure that non-`RandomVariable` `MeasurableVariable`s can be transformed.

    This test is specific to `MixtureRV`, which is derived from an `OpFromGraph`.
    """

    I_rv = pt.random.bernoulli(0.5, name="I")
    Y_1_rv = pt.random.beta(100, 1, name="Y_1")
    Y_2_rv = pt.random.beta(1, 100, name="Y_2")

    # A `MixtureRV`, which is an `OpFromGraph` subclass, will replace this
    # `pt.stack` in the graph
    Y_rv = pt.stack([Y_1_rv, Y_2_rv])[I_rv]
    Y_rv.name = "Y"

    i_vv = I_rv.clone()
    i_vv.name = "i"
    y_vv = Y_rv.clone()
    y_vv.name = "y"

    logp_no_trans = conditional_logp(
        {Y_rv: y_vv, I_rv: i_vv},
    )
    logp_no_trans_comb = pt.sum([pt.sum(factor) for factor in logp_no_trans.values()])

    transform_rewrite = TransformValuesRewrite({y_vv: LogTransform()})

    logp_trans = conditional_logp(
        {Y_rv: y_vv, I_rv: i_vv},
        extra_rewrites=transform_rewrite,
        use_jacobian=False,
    )
    logp_trans_combined = pt.sum([pt.sum(factor) for factor in logp_trans.values()])

    # The untransformed graph should be the same as the transformed graph after
    # replacing the `Y_rv` value variable with a transformed version of itself
    logp_nt_fg = FunctionGraph(outputs=[logp_no_trans_comb], clone=False)
    y_trans = pt.exp(y_vv)
    y_trans.name = "y_log"
    logp_nt_fg.replace(y_vv, y_trans)
    logp_nt = logp_nt_fg.outputs[0]

    assert equal_computations([logp_nt], [logp_trans_combined])


def test_scan_transform():
    """Test that Scan valued variables can be transformed"""

    init = pt.random.beta(1, 1, name="init")
    init_vv = init.clone()

    def scan_step(prev_innov):
        next_innov = pt.random.beta(prev_innov * 10, (1 - prev_innov) * 10)
        update = {next_innov.owner.inputs[0]: next_innov.owner.outputs[0]}
        return next_innov, update

    innov, _ = scan(
        fn=scan_step,
        outputs_info=[init],
        n_steps=4,
    )
    innov.name = "innov"
    innov_vv = innov.clone()

    tr = TransformValuesRewrite(
        {
            init_vv: LogOddsTransform(),
            innov_vv: LogOddsTransform(),
        }
    )
    logp = conditional_logp({init: init_vv, innov: innov_vv}, extra_rewrites=tr, use_jacobian=True)[
        innov_vv
    ]
    logp_fn = pytensor.function([init_vv, innov_vv], logp, on_unused_input="ignore")

    # Create an unrolled scan graph as reference
    innov = []
    prev_innov = init
    for i in range(4):
        next_innov = pt.random.beta(prev_innov * 10, (1 - prev_innov) * 10, name="innov[i]")
        innov.append(next_innov)
        prev_innov = next_innov
    innov = pt.stack(innov)
    innov.name = "innov"

    tr = TransformValuesRewrite(
        {
            init_vv: LogOddsTransform(),
            innov_vv: LogOddsTransform(),
        }
    )
    ref_logp = conditional_logp(
        {init: init_vv, innov: innov_vv}, extra_rewrites=tr, use_jacobian=True
    )[innov_vv]
    ref_logp_fn = pytensor.function([init_vv, innov_vv], ref_logp, on_unused_input="ignore")

    test_point = {
        "init": np.array(-0.5),
        "innov": np.full((4,), -0.5),
    }
    np.testing.assert_allclose(logp_fn(**test_point), ref_logp_fn(**test_point))


def test_weakref_leak():
    """Check that the rewrite does not have a growing memory footprint.

    See #6990
    """

    def _growth(limit=10, peak_stats={}):
        """Vendoring of objgraph.growth

        Source: https://github.com/mgedmin/objgraph/blob/94b1ca61a11109547442701800292dcfc7f59fc8/objgraph.py#L253
        """
        gc.collect()
        objects = gc.get_objects()

        stats = {}
        for o in objects:
            n = type(o).__name__
            stats[n] = stats.get(n, 0) + 1

        deltas = {}
        for name, count in stats.items():
            old_count = peak_stats.get(name, 0)
            if count > old_count:
                deltas[name] = count - old_count
                peak_stats[name] = count

        deltas = sorted(deltas.items(), key=operator.itemgetter(1), reverse=True)

        if limit:
            deltas = deltas[:limit]

        return [(name, stats[name], delta) for name, delta in deltas]

    rvs_to_values = {pt.random.beta(1, 1, name=f"p_{i}"): pt.scalar(f"p_{i}") for i in range(30)}
    tr = TransformValuesRewrite({v: logodds for v in rvs_to_values.values()})

    for i in range(20):
        conditional_logp(rvs_to_values, extra_rewrites=tr)
        res = _growth()
        # Only start checking after warmup
        if i > 15:
            assert not res, "Object counts are still growing"
