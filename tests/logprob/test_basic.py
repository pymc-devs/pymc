#   Copyright 2024 - present The PyMC Developers
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

import warnings

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats.distributions as sp

from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import equal_computations
from pytensor.graph.rewriting.basic import SequentialGraphRewriter, in2out, node_rewriter
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.op import RandomVariable
from scipy import stats

import pymc as pm

from pymc.logprob.abstract import MeasurableOp, ValuedRV, _logcdf, _logprob
from pymc.logprob.basic import (
    conditional_logp,
    icdf,
    logccdf,
    logcdf,
    logp,
    transformed_conditional_logp,
)
from pymc.logprob.rewriting import logprob_rewrites_basic_query, logprob_rewrites_db
from pymc.logprob.transforms import LogTransform
from pymc.logprob.utils import replace_rvs_by_values
from pymc.testing import assert_no_rvs


def test_factorized_joint_logprob_basic():
    # A simple check for when `factorized_joint_logprob` is the same as `logprob`
    a = pt.random.uniform(0.0, 1.0)
    a.name = "a"
    a_value_var = a.clone()

    a_logp = conditional_logp({a: a_value_var})
    a_logp_comb = next(iter(a_logp.values()))
    a_logp_exp = logp(a, a_value_var)

    assert equal_computations([a_logp_comb], [a_logp_exp])

    # Let's try a hierarchical model
    sigma = pt.random.invgamma(0.5, 0.5)
    Y = pt.random.normal(0.0, sigma)

    sigma_value_var = sigma.clone()
    y_value_var = Y.clone()

    total_ll = conditional_logp({Y: y_value_var, sigma: sigma_value_var})
    total_ll_combined = pt.add(*total_ll.values())

    # We need to replace the reference to `sigma` in `Y` with its value
    # variable
    ll_Y = logp(Y, y_value_var)
    (ll_Y,) = replace_rvs_by_values(
        [ll_Y],
        rvs_to_values={sigma: sigma_value_var},
    )
    total_ll_exp = ll_Y + logp(sigma, sigma_value_var)

    assert equal_computations([total_ll_combined], [total_ll_exp])

    # Now, make sure we can compute a joint log-probability for a hierarchical
    # model with some non-`RandomVariable` nodes
    c = pt.random.normal()
    c.name = "c"
    b_l = c * a + 2.0
    b = pt.random.uniform(b_l, b_l + 1.0)
    b.name = "b"

    b_value_var = b.clone()
    c_value_var = c.clone()

    b_logp = conditional_logp({a: a_value_var, b: b_value_var, c: c_value_var})
    b_logp_combined = pt.sum([pt.sum(factor) for factor in b_logp.values()])

    # There shouldn't be any `RandomVariable`s in the resulting graph
    assert_no_rvs(b_logp_combined)

    res_ancestors = list(ancestors((b_logp_combined,)))
    assert b_value_var in res_ancestors
    assert c_value_var in res_ancestors
    assert a_value_var in res_ancestors


def test_factorized_joint_logprob_multi_obs():
    a = pt.random.uniform(0.0, 1.0)
    b = pt.random.normal(0.0, 1.0)

    a_val = a.clone()
    b_val = b.clone()

    logp_res = conditional_logp({a: a_val, b: b_val})
    logp_res_combined = pt.add(*logp_res.values())
    logp_exp = logp(a, a_val) + logp(b, b_val)

    assert equal_computations([logp_res_combined], [logp_exp])

    x = pt.random.normal(0, 1)
    y = pt.random.normal(x, 1)

    x_val = x.clone()
    y_val = y.clone()

    logp_res = conditional_logp({x: x_val, y: y_val})
    exp_logp = conditional_logp({x: x_val, y: y_val})
    logp_res_comb = pt.sum([pt.sum(factor) for factor in logp_res.values()])
    exp_logp_comb = pt.sum([pt.sum(factor) for factor in exp_logp.values()])

    assert equal_computations([logp_res_comb], [exp_logp_comb])


def test_factorized_joint_logprob_diff_dims():
    M = pt.matrix("M")
    x = pt.random.normal(0, 1, size=M.shape[1], name="X")
    y = pt.random.normal(M.dot(x), 1, name="Y")

    x_vv = x.clone()
    x_vv.name = "x"
    y_vv = y.clone()
    y_vv.name = "y"

    logp = conditional_logp({x: x_vv, y: y_vv})
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

    M_val = np.random.normal(size=(10, 3))
    x_val = np.random.normal(size=(3,))
    y_val = np.random.normal(size=(10,))

    point = {M: M_val, x_vv: x_val, y_vv: y_val}
    logp_val = logp_combined.eval(point)

    exp_logp_val = (
        sp.norm.logpdf(x_val, 0, 1).sum() + sp.norm.logpdf(y_val, M_val.dot(x_val), 1).sum()
    )
    assert exp_logp_val == pytest.approx(logp_val)


def test_persist_inputs():
    """Make sure we don't unnecessarily clone variables."""
    x = pt.scalar("x")
    beta_rv = pt.random.normal(0, 1, name="beta")
    Y_rv = pt.random.normal(beta_rv * x, 1, name="y")

    beta_vv = beta_rv.type()
    y_vv = Y_rv.clone()

    logp = conditional_logp({beta_rv: beta_vv, Y_rv: y_vv})
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

    assert x in ancestors([logp_combined])

    # Make sure we don't clone value variables when they're graphs.
    y_vv_2 = y_vv * 2
    logp_2 = conditional_logp({beta_rv: beta_vv, Y_rv: y_vv_2})
    logp_2_combined = pt.sum([pt.sum(factor) for factor in logp_2.values()])

    assert y_vv in ancestors([logp_2_combined])
    assert y_vv_2 in ancestors([logp_2_combined])

    # Even when they are random
    y_vv = pt.random.normal(name="y_vv2")
    y_vv_2 = y_vv * 2
    logp_2 = conditional_logp({beta_rv: beta_vv, Y_rv: y_vv_2})
    logp_2_combined = pt.sum([pt.sum(factor) for factor in logp_2.values()])

    assert y_vv in ancestors([logp_2_combined])
    assert y_vv_2 in ancestors([logp_2_combined])


def test_warn_rvs_conditional_logp():
    x_rv = pt.random.normal(name="x")
    y_rv = pt.random.normal(x_rv, 1, name="y")

    y_vv = y_rv.clone()

    with pytest.warns(UserWarning, match="Random variables detected in the logp graph: {x}"):
        conditional_logp({y_rv: y_vv})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        conditional_logp({y_rv: y_vv}, warn_rvs=False)


def test_multiple_rvs_to_same_value_raises():
    x_rv1 = pt.random.normal(name="x1")
    x_rv2 = pt.random.normal(name="x2")
    x = x_rv1.type()
    x.name = "x"

    msg = "More than one logprob term was assigned to the value var x"
    with pytest.raises(ValueError, match=msg):
        conditional_logp({x_rv1: x, x_rv2: x})


def test_joint_logp_basic():
    """Make sure we can compute a log-likelihood for a hierarchical model with transforms."""

    with pm.Model() as m:
        a = pm.Uniform("a", 0.0, 1.0)
        c = pm.Normal("c")
        b_l = c * a + 2.0
        b = pm.Uniform("b", b_l, b_l + 1.0)

    a_value_var = m.rvs_to_values[a]
    assert m.rvs_to_transforms[a]

    b_value_var = m.rvs_to_values[b]
    assert m.rvs_to_transforms[b]

    c_value_var = m.rvs_to_values[c]

    (b_logp,) = transformed_conditional_logp(
        (b,),
        rvs_to_values=m.rvs_to_values,
        rvs_to_transforms=m.rvs_to_transforms,
    )

    # There shouldn't be any `RandomVariable`s in the resulting graph
    assert_no_rvs(b_logp)

    res_ancestors = list(ancestors((b_logp,)))
    assert b_value_var in res_ancestors
    assert c_value_var in res_ancestors
    assert a_value_var in res_ancestors


def test_joint_logp_over_multiple_values():
    """An op whose logp is joint over several values must receive all of them in one call.

    Each value used to be recursed on its own as soon as a measurable chain reached it, so a
    node valued through a chain had its density silently derived over a subset of its values.
    The values arrive in output order, and a subset is only interpretable if the op knows which
    outputs it belongs to, so the rewrite that makes the op measurable records which of them
    lead to a value.
    """

    class JointPairOp(OpFromGraph):
        """Not yet measurable; `find_measurable_joint_pair` makes it so."""

    class MeasurableJointPairOp(MeasurableOp, OpFromGraph):
        # The first value's density is over its trailing axis; the second's is elementwise
        supp_axes = ((-1,), ())
        measured_outputs: tuple[bool, bool]

    def variables_leading_to_values(fgraph):
        # Like the pymc util of the same name, but through clients of any kind: at rewrite time
        # the chains above this op have not yet been made measurable
        leads = set()
        for node in reversed(fgraph.toposort()):
            if isinstance(node.op, ValuedRV) or any(out in leads for out in node.outputs):
                leads.update(node.inputs)
        return leads

    @node_rewriter([JointPairOp])
    def find_measurable_joint_pair(fgraph, node):
        inner_inputs = [inp.type() for inp in node.op.fgraph.inputs]
        measurable_op = MeasurableJointPairOp(
            inputs=inner_inputs, outputs=node.op.fgraph.bind(inner_inputs)
        )
        leads_to_value = variables_leading_to_values(fgraph)
        measurable_op.measured_outputs = tuple(out in leads_to_value for out in node.outputs)
        return measurable_op(*node.inputs, return_list=True)

    ir_rewriter = SequentialGraphRewriter(
        in2out(find_measurable_joint_pair),
        logprob_rewrites_db.query(logprob_rewrites_basic_query),
    )

    calls = []

    @_logprob.register(MeasurableJointPairOp)
    def joint_pair_logp(op, values, *inputs, **kwargs):
        calls.append((op.measured_outputs, len(values)))
        assert len(values) == sum(op.measured_outputs), "logp derived over a subset of values"
        if op.measured_outputs == (True, True):
            v1, v2 = values
            # Deliberately non-separable: the whole joint term is assigned to the first value,
            # and the second gets a placeholder shaped like a real term would be.
            return v1.sum(axis=-1) + v2, pt.zeros_like(v2)
        elif op.measured_outputs == (True, False):
            [v1] = values
            return v1.sum(axis=-1) + 100.0
        else:
            [v2] = values
            return v2 - 100.0

    @_logcdf.register(MeasurableJointPairOp)
    def joint_pair_logcdf(op, value, *inputs, **kwargs):
        # Censoring derives the base's marginal logcdf eagerly; the bounds lie outside the
        # test values, so this term never survives into the evaluated branches
        return pt.zeros_like(value)

    mu = pt.vector("mu", shape=(3,))
    op = JointPairOp(inputs=[mu], outputs=[pt.broadcast_to(mu[:, None], (3, 5)), mu * 2])
    v1 = pt.matrix("v1", shape=(3, 5))
    v2 = pt.vector("v2", shape=(3,))
    v1_test = np.arange(15, dtype=v1.dtype).reshape(3, 5)
    v2_test = np.array([1.0, 2.0, 3.0], dtype=v2.dtype)

    # Both outputs measured: one value arrives through a measurable chain, the other directly
    out1, out2 = op(mu)
    logps = conditional_logp({out1 + 1: v1, out2: v2}, ir_rewriter=ir_rewriter)
    assert calls == [((True, True), 2)]
    fn = pytensor.function([v1, v2], [logps[v1], logps[v2]])
    got1, got2 = fn(v1_test, v2_test)
    np.testing.assert_allclose(got1, (v1_test - 1).sum(axis=-1) + v2_test)
    np.testing.assert_allclose(got2, np.zeros(3))

    # Both outputs measured, each through its own chain, one of them censoring: no value is
    # attached to the joint node itself, and clip's logp must also reach it through the helper.
    # The clipped value passes through unchanged and lies inside the bounds, so its term is the
    # joint one untouched by the censoring branches.
    calls.clear()
    out1, out2 = op(mu)
    logps = conditional_logp(
        {out1 + 1: v1, pt.clip(out2, -1000.0, 1000.0): v2}, ir_rewriter=ir_rewriter
    )
    assert calls == [((True, True), 2)]
    fn = pytensor.function([v1, v2], [logps[v1], logps[v2]])
    got1, got2 = fn(v1_test, v2_test)
    np.testing.assert_allclose(got1, (v1_test - 1).sum(axis=-1) + v2_test)
    np.testing.assert_allclose(got2, np.zeros(3))

    # Only the first output measured, through a chain
    calls.clear()
    out1, _ = op(mu)
    [logp_v1] = conditional_logp({out1 + 1: v1}, ir_rewriter=ir_rewriter).values()
    assert calls == [((True, False), 1)]
    np.testing.assert_allclose(logp_v1.eval({v1: v1_test}), (v1_test - 1).sum(axis=-1) + 100.0)

    # Only the second output measured, through a chain
    calls.clear()
    _, out2 = op(mu)
    [logp_v2] = conditional_logp({out2 + 1: v2}, ir_rewriter=ir_rewriter).values()
    assert calls == [((False, True), 1)]
    np.testing.assert_allclose(logp_v2.eval({v2: v2_test}), (v2_test - 1) - 100.0)


def test_model_unchanged_logprob_access():
    # Issue #5007
    with pm.Model() as model:
        a = pm.Normal("a")
        c = pm.Uniform("c", lower=a - 1, upper=1)

    original_inputs = set(pytensor.graph.graph_inputs([c]))
    # Extract model.logp
    model.logp()
    new_inputs = set(pytensor.graph.graph_inputs([c]))
    assert original_inputs == new_inputs


def test_unexpected_rvs():
    with pm.Model() as model:
        x = pm.Normal("x")
        y = pm.CustomDist("y", logp=lambda *args: x)

    with pytest.raises(ValueError, match="^Random variables detected in the logp graph"):
        model.logp()


def test_hierarchical_logp():
    """Make sure there are no random variables in a model's log-likelihood graph."""
    with pm.Model() as m:
        x = pm.Uniform("x", lower=0, upper=1)
        y = pm.Uniform("y", lower=0, upper=x)

    logp_ancestors = list(ancestors([m.logp()]))
    ops = {a.owner.op for a in logp_ancestors if a.owner}
    assert len(ops) > 0
    assert not any(isinstance(o, RandomVariable) for o in ops)
    assert m.rvs_to_values[x] in logp_ancestors
    assert m.rvs_to_values[y] in logp_ancestors


def test_hierarchical_obs_logp():
    obs = np.array([0.5, 0.4, 5, 2])

    with pm.Model() as model:
        x = pm.Uniform("x", 0, 1, observed=obs)
        pm.Uniform("y", x, 2, observed=obs)

    logp_ancestors = list(ancestors([model.logp()]))
    ops = {a.owner.op for a in logp_ancestors if a.owner}
    assert len(ops) > 0
    assert not any(isinstance(o, RandomVariable) for o in ops)


@pytest.mark.parametrize(
    "func, scipy_func",
    [
        (logp, "logpdf"),
        (logcdf, "logcdf"),
        (icdf, "ppf"),
    ],
)
def test_probability_direct_dispatch(func, scipy_func):
    value = pt.vector("value")
    x = pm.Normal.dist(0, 1)

    np.testing.assert_almost_equal(
        func(x, value).eval({value: [0, 1]}),
        getattr(sp.norm(0, 1), scipy_func)([0, 1]),
    )

    np.testing.assert_almost_equal(
        func(x, [0, 1]).eval(),
        getattr(sp.norm(0, 1), scipy_func)([0, 1]),
    )


@pytest.mark.parametrize(
    "func, scipy_func, test_value",
    [
        (logp, "logpdf", 5.0),
        (logcdf, "logcdf", 5.0),
        (logccdf, "logccdf", 5.0),
        (icdf, "ppf", 0.7),
    ],
)
def test_probability_inference(func, scipy_func, test_value):
    if scipy_func == "logccdf":
        # Scipy lognorm doesn't have logccdf
        expected = np.log1p(-sp.lognorm(s=1).cdf(test_value))
    else:
        expected = getattr(sp.lognorm(s=1), scipy_func)(test_value)

    res = func(pt.exp(pm.Normal.dist()), test_value).eval()
    assert res.shape == ()
    np.testing.assert_allclose(res, expected)

    res = func(pt.exp(pm.Normal.dist(size=(2,))), test_value).eval()
    assert res.shape == (2,)
    np.testing.assert_allclose(res, expected)

    res = func(pt.exp(pm.Normal.dist(size=(2,))), np.broadcast_to(test_value, (3, 2))).eval()
    assert res.shape == (3, 2)
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize(
    "func, func_name",
    [
        (logp, "Logprob"),
        (logcdf, "LogCDF"),
        (icdf, "Inverse CDF"),
    ],
)
def test_probability_inference_fails(func, func_name):
    with pytest.raises(
        NotImplementedError,
        match=f"{func_name} method not implemented for (Elemwise{{cos,no_inplace}}|Cos)",
    ):
        func(pt.cos(pm.Normal.dist()), 1)


@pytest.mark.parametrize(
    "func, scipy_func, test_value",
    [
        (logp, "logpdf", 5.0),
        (logcdf, "logcdf", 5.0),
        (icdf, "ppf", 0.7),
    ],
)
def test_warn_rvs_probability_derivation(func, scipy_func, test_value):
    # Fail if unexpected warning is issued
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        input_rv = pm.Normal.dist(0, name="input")
        # Note: This graph could correspond to a convolution of two normals
        # In which case the inference should either return that or fail explicitly
        # For now, the lopgrob submodule treats the input as a stochastic value.
        rv = pt.exp(pm.Normal.dist(input_rv))
        with pytest.warns(
            UserWarning, match="RandomVariables {input} were found in the derived graph"
        ):
            func(rv, 0.0)

        res = func(rv, 0.0, warn_rvs=False)
        # This is the problem we are warning about, as now we can no longer identify the original rv in the graph
        # or replace it by the respective value
        assert rv not in ancestors([res])

        # Test that the prescribed solution does not raise a warning and works as expected
        input_vv = input_rv.clone()
        [new_rv] = replace_rvs_by_values(
            [rv],
            rvs_to_values={input_rv: input_vv},
            rvs_to_transforms={input_rv: LogTransform()},
        )
        input_vv_test = 1.3
        np.testing.assert_almost_equal(
            func(new_rv, test_value).eval({input_vv: input_vv_test}),
            getattr(sp.lognorm(s=1, loc=0, scale=np.exp(np.exp(input_vv_test))), scipy_func)(
                test_value
            ),
        )


def test_icdf_discrete():
    p = 0.1
    value = 0.9
    dist = pm.Geometric.dist(p=p)
    dist_icdf = icdf(dist, value)
    np.testing.assert_almost_equal(
        dist_icdf.eval(),
        sp.geom.ppf(value, p),
    )


def test_ir_rewrite_does_not_disconnect_valued_rvs():
    """Check that we don't lose the dependency across RV values do to automatic rewrites.

    See ValuedRV docstrings for more context.

    Regression test for https://github.com/pymc-devs/pymc/issues/6917
    """
    a_base = pm.Normal.dist()
    a = a_base * 5
    b = pm.Normal.dist(a * 8)

    a_value = a.type()
    b_value = b.type()
    logp_b = conditional_logp({a: a_value, b: b_value})[b_value]

    assert_no_rvs(logp_b)
    np.testing.assert_allclose(
        logp_b.eval({a_value: np.pi, b_value: np.e}),
        stats.norm.logpdf(np.e, np.pi * 8, 1),
    )


def test_ir_ops_can_be_evaluated_with_warning():
    _eval_values = [None, None]

    def my_logp(value, lam):
        nonlocal _eval_values
        _eval_values[0] = value.eval()
        _eval_values[1] = lam.eval({"lam_log__": -1.5})
        return value * lam

    with pm.Model() as m:
        lam = pm.Exponential("lam")
        pm.CustomDist("y", lam, logp=my_logp, observed=[0, 1, 2])

    # A dependency reaches the logp as its value, not as the ValuedRV standing for it, so the
    # only IR op left here is the TransformedValue one (see #8100).
    with pytest.warns(
        UserWarning, match="TransformedValue should not be present in the final graph"
    ):
        m.logp()

    assert _eval_values[0].sum() == 3
    assert _eval_values[1] == np.exp(-1.5)


def test_broadcasted_logp_does_not_reference_rv():
    """Size referencing another RV's shape should not leak into the logp graph.

    RVs in distribution parameters (e.g. Normal(mu=rv)) are expected — they
    represent statistical dependencies. But RVs in the size are shape artifacts
    from broadcast_shape(rv, ...) that have no role in the logp computation.
    """
    rv1 = pm.Normal.dist([0, 1, 2])
    rv2 = pm.Normal.dist(0, 1, size=rv1.shape)
    assert_no_rvs(pm.logp(rv2, 0))

    # Faithful regression case for #8301
    def logp(value, sigma):
        inner = pm.Truncated.dist(pm.Normal.dist(sigma=sigma), lower=-5.0, upper=5.0)
        outer = pm.Truncated.dist(inner, lower=0.1)
        return pm.logp(outer, value)

    with pm.Model() as m:
        pm.CustomDist("x", np.ones(3), logp=logp)

    assert_no_rvs(m.logp())
