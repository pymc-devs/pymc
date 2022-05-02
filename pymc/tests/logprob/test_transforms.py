#   Copyright 2022- The PyMC Developers
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
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import numpy as np
import pytensor
import pytensor.tensor as at
import pytest
import scipy as sp
import scipy.special

from numdifftools import Jacobian
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.scan import scan

from pymc.distributions.transforms import _default_transform, log, logodds
from pymc.logprob.abstract import MeasurableVariable, _get_measurable_outputs, _logprob
from pymc.logprob.joint_logprob import factorized_joint_logprob, joint_logprob
from pymc.logprob.transforms import (
    ChainedTransform,
    ExpTransform,
    IntervalTransform,
    LocTransform,
    LogOddsTransform,
    LogTransform,
    RVTransform,
    ScaleTransform,
    TransformValuesMapping,
    TransformValuesRewrite,
    transformed_variable,
)
from pymc.tests.helpers import assert_no_rvs


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
            scipy.special.xlogy(self.alphas - 1, value) - scipy.special.gammaln(self.alphas),
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
        pytest.param(
            at.random.wald,
            (1.5, 10.5),
            lambda mean, scale: sp.stats.invgauss(mean / scale, scale=scale),
            (),
            marks=pytest.mark.xfail(
                reason="We don't use PyTensor's Wald operator",
                raises=NotImplementedError,
            ),
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
        pytest.param(
            at.random.weibull,
            (1.5,),
            lambda c: sp.stats.weibull_min(c),
            (),
            marks=pytest.mark.xfail(
                reason="We don't use PyTensor's Weibull operator",
                raises=NotImplementedError,
            ),
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

    TODO: This test is rather redundant with those in tess/distributions/test_transform.py
    """

    a = at_dist(*dist_params, size=size)
    a.name = "a"
    a_value_var = at.tensor(a.dtype, shape=(None,) * a.ndim)
    a_value_var.name = "a_value"

    b = at.random.normal(a, 1.0)
    b.name = "b"
    b_value_var = b.clone()
    b_value_var.name = "b_value"

    transform = _default_transform(a.owner.op, a)
    transform_rewrite = TransformValuesRewrite({a_value_var: transform})
    res = joint_logprob({a: a_value_var, b: b_value_var}, extra_rewrites=transform_rewrite)

    test_val_rng = np.random.RandomState(3238)

    logp_vals_fn = pytensor.function([a_value_var, b_value_var], res)

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

    transform_rewrite = TransformValuesRewrite({x_vv: log})
    tr_logp = joint_logprob(
        {X_rv: x_vv}, extra_rewrites=transform_rewrite, use_jacobian=use_jacobian
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

    transform_rewrite = TransformValuesRewrite(
        {
            lower: _default_transform(lower_rv.owner.op, lower_rv),
            upper: _default_transform(upper_rv.owner.op, upper_rv),
            x: _default_transform(x_rv.owner.op, x_rv),
        }
    )
    logp = joint_logprob(
        {lower_rv: lower, upper_rv: upper, x_rv: x},
        extra_rewrites=transform_rewrite,
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

    transform_rewrite = TransformValuesRewrite(
        {
            loc: None,
            scale: LogOddsTransform(),
            x: LogTransform(),
        }
    )

    logp = joint_logprob(
        {loc_rv: loc, scale_rv: scale, x_rv: x},
        extra_rewrites=transform_rewrite,
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

    transform_rewrite = TransformValuesRewrite({x: None})

    logp = joint_logprob(
        {x_rv: x},
        extra_rewrites=transform_rewrite,
    )

    assert np.isclose(
        logp.eval({x: 1}),
        sp.stats.norm(0, 1).logpdf(1),
    )


@pytest.fixture(scope="module")
def multiout_measurable_op():
    # Create a dummy Op that just returns the two inputs
    mu1, mu2 = at.scalars("mu1", "mu2")

    class TestOpFromGraph(OpFromGraph):
        def do_constant_folding(self, fgraph, node):
            False

    multiout_op = TestOpFromGraph([mu1, mu2], [mu1 + 0.0, mu2 + 0.0])

    MeasurableVariable.register(TestOpFromGraph)

    @_logprob.register(TestOpFromGraph)
    def logp_multiout(op, values, mu1, mu2):
        value1, value2 = values
        return value1 + mu1, value2 + mu2

    @_get_measurable_outputs.register(TestOpFromGraph)
    def measurable_multiout_op_outputs(op, node):
        return node.outputs

    return multiout_op


@pytest.mark.parametrize("transform_x", (True, False))
@pytest.mark.parametrize("transform_y", (True, False))
def test_nondefault_transform_multiout(transform_x, transform_y, multiout_measurable_op):
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

    logp = joint_logprob({x: x_vv, y: y_vv}, extra_rewrites=transform_rewrite)

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

    np.testing.assert_almost_equal(logp.eval({x_vv: x_vv_test, y_vv: y_vv_test}), expected_logp)


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

    tr = TransformValuesRewrite({p_vv: logodds})
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

    transform_rewrite = TransformValuesRewrite({y_vv: LogTransform()})

    with pytest.warns(None) as record:
        # This shouldn't raise any warnings
        logp_trans = joint_logprob(
            {Y_rv: y_vv, I_rv: i_vv},
            extra_rewrites=transform_rewrite,
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


def test_exp_transform_rv():
    base_rv = at.random.normal(0, 1, size=2, name="base_rv")
    y_rv = at.exp(base_rv)
    y_rv.name = "y"

    y_vv = y_rv.clone()
    logp = joint_logprob({y_rv: y_vv}, sum=False)
    logp_fn = pytensor.function([y_vv], logp)

    y_val = [0.1, 0.3]
    np.testing.assert_allclose(
        logp_fn(y_val),
        sp.stats.lognorm(s=1).logpdf(y_val),
    )


def test_log_transform_rv():
    base_rv = at.random.lognormal(0, 1, size=2, name="base_rv")
    y_rv = at.log(base_rv)
    y_rv.name = "y"

    y_vv = y_rv.clone()
    logp = joint_logprob({y_rv: y_vv}, sum=False)
    logp_fn = pytensor.function([y_vv], logp)

    y_val = [0.1, 0.3]
    np.testing.assert_allclose(
        logp_fn(y_val),
        sp.stats.norm().logpdf(y_val),
    )


@pytest.mark.parametrize(
    "rv_size, loc_type, addition",
    [
        (None, at.scalar, True),
        (2, at.vector, False),
        ((2, 1), at.col, True),
    ],
)
def test_loc_transform_rv(rv_size, loc_type, addition):

    loc = loc_type("loc")
    if addition:
        y_rv = loc + at.random.normal(0, 1, size=rv_size, name="base_rv")
    else:
        y_rv = at.random.normal(0, 1, size=rv_size, name="base_rv") - at.neg(loc)
    y_rv.name = "y"
    y_vv = y_rv.clone()

    logp = joint_logprob({y_rv: y_vv}, sum=False)
    assert_no_rvs(logp)
    logp_fn = pytensor.function([loc, y_vv], logp)

    loc_test_val = np.full(rv_size, 4.0)
    y_test_val = np.full(rv_size, 1.0)

    np.testing.assert_allclose(
        logp_fn(loc_test_val, y_test_val),
        sp.stats.norm(loc_test_val, 1).logpdf(y_test_val),
    )


@pytest.mark.parametrize(
    "rv_size, scale_type, product",
    [
        (None, at.scalar, True),
        (1, at.TensorType("floatX", (True,)), True),
        ((2, 3), at.matrix, False),
    ],
)
def test_scale_transform_rv(rv_size, scale_type, product):

    scale = scale_type("scale")
    if product:
        y_rv = at.random.normal(0, 1, size=rv_size, name="base_rv") * scale
    else:
        y_rv = at.random.normal(0, 1, size=rv_size, name="base_rv") / at.reciprocal(scale)
    y_rv.name = "y"
    y_vv = y_rv.clone()

    logp = joint_logprob({y_rv: y_vv}, sum=False)
    assert_no_rvs(logp)
    logp_fn = pytensor.function([scale, y_vv], logp)

    scale_test_val = np.full(rv_size, 4.0)
    y_test_val = np.full(rv_size, 1.0)

    np.testing.assert_allclose(
        logp_fn(scale_test_val, y_test_val),
        sp.stats.norm(0, scale_test_val).logpdf(y_test_val),
    )


def test_transformed_rv_and_value():
    y_rv = at.random.halfnormal(-1, 1, name="base_rv") + 1
    y_rv.name = "y"
    y_vv = y_rv.clone()

    transform_rewrite = TransformValuesRewrite({y_vv: LogTransform()})

    logp = joint_logprob({y_rv: y_vv}, extra_rewrites=transform_rewrite)
    assert_no_rvs(logp)
    logp_fn = pytensor.function([y_vv], logp)

    y_test_val = -5

    assert np.isclose(
        logp_fn(y_test_val),
        sp.stats.halfnorm(0, 1).logpdf(np.exp(y_test_val)) + y_test_val,
    )


def test_loc_transform_multiple_rvs_fails1():
    x_rv1 = at.random.normal(name="x_rv1")
    x_rv2 = at.random.normal(name="x_rv2")
    y_rv = x_rv1 + x_rv2

    y = y_rv.clone()

    with pytest.raises(RuntimeError, match="could not be derived"):
        joint_logprob({y_rv: y})


def test_nested_loc_transform_multiple_rvs_fails2():
    x_rv1 = at.random.normal(name="x_rv1")
    x_rv2 = at.cos(at.random.normal(name="x_rv2"))
    y_rv = x_rv1 + x_rv2

    y = y_rv.clone()

    with pytest.raises(RuntimeError, match="could not be derived"):
        joint_logprob({y_rv: y})


def test_discrete_rv_unary_transform_fails():
    y_rv = at.exp(at.random.poisson(1))
    with pytest.raises(RuntimeError, match="could not be derived"):
        joint_logprob({y_rv: y_rv.clone()})


def test_discrete_rv_multinary_transform_fails():
    y_rv = 5 + at.random.poisson(1)
    with pytest.raises(RuntimeError, match="could not be derived"):
        joint_logprob({y_rv: y_rv.clone()})


@pytest.mark.xfail(reason="Check not implemented yet, see #51")
def test_invalid_broadcasted_transform_rv_fails():
    loc = at.vector("loc")
    y_rv = loc + at.random.normal(0, 1, size=2, name="base_rv")
    y_rv.name = "y"
    y_vv = y_rv.clone()

    logp = joint_logprob({y_rv: y_vv})
    logp.eval({y_vv: [0, 0, 0, 0], loc: [0, 0, 0, 0]})
    assert False, "Should have failed before"


@pytest.mark.parametrize("numerator", (1.0, 2.0))
def test_reciprocal_rv_transform(numerator):
    shape = 3
    scale = 5
    x_rv = numerator / at.random.gamma(shape, scale)
    x_rv.name = "x"

    x_vv = x_rv.clone()
    x_logp_fn = pytensor.function([x_vv], joint_logprob({x_rv: x_vv}))

    x_test_val = 1.5
    assert np.isclose(
        x_logp_fn(x_test_val),
        sp.stats.invgamma(shape, scale=scale * numerator).logpdf(x_test_val),
    )


def test_negated_rv_transform():
    x_rv = -at.random.halfnormal()
    x_rv.name = "x"

    x_vv = x_rv.clone()
    x_logp_fn = pytensor.function([x_vv], joint_logprob({x_rv: x_vv}))

    assert np.isclose(x_logp_fn(-1.5), sp.stats.halfnorm.logpdf(1.5))


def test_subtracted_rv_transform():
    # Choose base RV that is assymetric around zero
    x_rv = 5.0 - at.random.normal(1.0)
    x_rv.name = "x"

    x_vv = x_rv.clone()
    x_logp_fn = pytensor.function([x_vv], joint_logprob({x_rv: x_vv}))

    assert np.isclose(x_logp_fn(7.3), sp.stats.norm.logpdf(5.0 - 7.3, 1.0))


def test_scan_transform():
    """Test that Scan valued variables can be transformed"""

    init = at.random.beta(1, 1, name="init")
    init_vv = init.clone()

    innov, _ = scan(
        fn=lambda prev_innov: at.random.beta(prev_innov * 10, (1 - prev_innov) * 10),
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
    logp = factorized_joint_logprob(
        {init: init_vv, innov: innov_vv}, extra_rewrites=tr, use_jacobian=True
    )[innov_vv]
    logp_fn = pytensor.function([init_vv, innov_vv], logp, on_unused_input="ignore")

    # Create an unrolled scan graph as reference
    innov = []
    prev_innov = init
    for i in range(4):
        next_innov = at.random.beta(prev_innov * 10, (1 - prev_innov) * 10, name=f"innov[i]")
        innov.append(next_innov)
        prev_innov = next_innov
    innov = at.stack(innov)
    innov.name = "innov"

    tr = TransformValuesRewrite(
        {
            init_vv: LogOddsTransform(),
            innov_vv: LogOddsTransform(),
        }
    )
    ref_logp = factorized_joint_logprob(
        {init: init_vv, innov: innov_vv}, extra_rewrites=tr, use_jacobian=True
    )[innov_vv]
    ref_logp_fn = pytensor.function([init_vv, innov_vv], ref_logp, on_unused_input="ignore")

    test_point = {
        "init": np.array(-0.5),
        "innov": np.full((4,), -0.5),
    }
    np.testing.assert_allclose(logp_fn(**test_point), ref_logp_fn(**test_point))
