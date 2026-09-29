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

import numpy as np
import pytensor
import pytest

from pytensor import tensor as pt
from pytensor.assumptions import assume
from pytensor.compile.ops import DeepCopyOp
from pytensor.graph import RewriteDatabaseQuery
from pytensor.tensor.random.type import random_generator_type
from scipy import stats as st

import pymc as pm

from pymc.logprob.basic import conditional_logp, icdf, logcdf, logp
from pymc.logprob.rewriting import logprob_rewrites_db
from pymc.testing import assert_no_rvs


def test_bcast_rv_logp():
    """Test that derived logp for broadcasted RV is correct"""

    x_rv = pt.random.normal(name="x")
    broadcasted_x_rv = pt.broadcast_to(x_rv, (2,))
    broadcasted_x_rv.name = "broadcasted_x"
    broadcasted_x_vv = broadcasted_x_rv.clone()

    logp = conditional_logp({broadcasted_x_rv: broadcasted_x_vv})
    logp_combined = pt.add(*logp.values())
    valid_logp = logp_combined.eval({broadcasted_x_vv: [0, 0]})

    # The broadcast dimension is consumed like a support dimension
    assert valid_logp.shape == ()
    assert np.isclose(valid_logp, st.norm.logpdf(0))

    # It's not possible for broadcasted dimensions to have different values
    invalid_logp = logp_combined.eval({broadcasted_x_vv: [0, 1]})
    assert invalid_logp == -np.inf


def test_measurable_make_vector():
    base1_rv = pt.random.normal(name="base1")
    base2_rv = pt.random.halfnormal(name="base2")
    base3_rv = pt.random.exponential(name="base3")
    y_rv = pt.stack((base1_rv, base2_rv, base3_rv))
    y_rv.name = "y"

    base1_vv = base1_rv.clone()
    base2_vv = base2_rv.clone()
    base3_vv = base3_rv.clone()
    y_vv = y_rv.clone()

    ref_logp = conditional_logp({base1_rv: base1_vv, base2_rv: base2_vv, base3_rv: base3_vv})
    ref_logp_combined = pt.sum([pt.sum(factor) for factor in ref_logp.values()])

    make_vector_logp = logp(y_rv, y_vv)

    base1_testval = base1_rv.eval()
    base2_testval = base2_rv.eval()
    base3_testval = base3_rv.eval()
    y_testval = np.stack((base1_testval, base2_testval, base3_testval))

    ref_logp_eval_eval = ref_logp_combined.eval(
        {base1_vv: base1_testval, base2_vv: base2_testval, base3_vv: base3_testval}
    )
    make_vector_logp_eval = make_vector_logp.eval({y_vv: y_testval})

    assert make_vector_logp_eval.shape == y_testval.shape
    assert np.isclose(make_vector_logp_eval.sum(), ref_logp_eval_eval)


def test_measurable_make_vector_with_constant_input():
    base1_rv = pt.random.normal(name="base1")
    base2_rv = pt.random.halfnormal(name="base2")
    y_rv = pt.stack((base1_rv, pt.constant(0.0), base2_rv))
    y_rv.name = "y"

    base1_vv = base1_rv.clone()
    base2_vv = base2_rv.clone()
    y_vv = y_rv.clone()

    ref_logp = conditional_logp({base1_rv: base1_vv, base2_rv: base2_vv})
    ref_logp_combined = pt.sum([pt.sum(factor) for factor in ref_logp.values()])
    y_logp = logp(y_rv, y_vv)

    base1_testval = base1_rv.eval()
    base2_testval = base2_rv.eval()
    y_testval = np.stack((base1_testval, 0.0, base2_testval)).astype(y_vv.dtype)

    ref_logp_eval = ref_logp_combined.eval({base1_vv: base1_testval, base2_vv: base2_testval})
    y_logp_eval = y_logp.eval({y_vv: y_testval})

    assert y_logp_eval.shape == y_testval.shape
    assert np.isclose(y_logp_eval.sum(), ref_logp_eval)

    y_testval_bad = y_testval.copy()
    y_testval_bad[1] = 1.0
    y_logp_eval_bad = y_logp.eval({y_vv: y_testval_bad})
    assert y_logp_eval_bad[1] == -np.inf


@pytest.mark.parametrize("reverse", (False, True))
def test_measurable_make_vector_interdependent(reverse):
    """Test that we can obtain a proper graph when stacked RVs depend on each other"""
    x = pt.random.normal(name="x")
    y_rvs = []
    prev_rv = x
    for i in range(3):
        next_rv = pt.random.normal(prev_rv + 1, name=f"y{i}")
        y_rvs.append(next_rv)
        prev_rv = next_rv

    if reverse:
        y_rvs = y_rvs[::-1]

    ys = pt.stack(y_rvs)
    ys.name = "ys"

    x_vv = x.clone()
    ys_vv = ys.clone()

    logp = conditional_logp({x: x_vv, ys: ys_vv})
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])
    assert_no_rvs(logp_combined)

    y0_vv = y_rvs[0].clone()
    y1_vv = y_rvs[1].clone()
    y2_vv = y_rvs[2].clone()

    ref_logp = conditional_logp({x: x_vv, y_rvs[0]: y0_vv, y_rvs[1]: y1_vv, y_rvs[2]: y2_vv})
    ref_logp_combined = pt.sum([pt.sum(factor) for factor in ref_logp.values()])

    rng = np.random.default_rng()
    x_vv_test = rng.normal()
    ys_vv_test = rng.normal(size=3)
    np.testing.assert_allclose(
        logp_combined.eval({x_vv: x_vv_test, ys_vv: ys_vv_test}).sum(),
        ref_logp_combined.eval(
            {x_vv: x_vv_test, y0_vv: ys_vv_test[0], y1_vv: ys_vv_test[1], y2_vv: ys_vv_test[2]}
        ),
    )


@pytest.mark.parametrize("reverse", (False, True))
def test_measurable_join_interdependent(reverse):
    """Test that we can obtain a proper graph when stacked RVs depend on each other"""
    x = pt.random.normal(name="x")
    y_rvs = []
    prev_rv = x
    for i in range(3):
        next_rv = pt.random.normal(prev_rv + 1, name=f"y{i}", size=(1, 2))
        y_rvs.append(next_rv)
        prev_rv = next_rv

    if reverse:
        y_rvs = y_rvs[::-1]

    ys = pt.concatenate(y_rvs, axis=0)
    ys.name = "ys"

    x_vv = x.clone()
    ys_vv = ys.clone()

    logp = conditional_logp({x: x_vv, ys: ys_vv})
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])
    assert_no_rvs(logp_combined)

    y0_vv = y_rvs[0].clone()
    y1_vv = y_rvs[1].clone()
    y2_vv = y_rvs[2].clone()

    ref_logp = conditional_logp({x: x_vv, y_rvs[0]: y0_vv, y_rvs[1]: y1_vv, y_rvs[2]: y2_vv})
    ref_logp_combined = pt.sum([pt.sum(factor) for factor in ref_logp.values()])

    rng = np.random.default_rng()
    x_vv_test = rng.normal()
    ys_vv_test = rng.normal(size=(3, 2))
    np.testing.assert_allclose(
        logp_combined.eval({x_vv: x_vv_test, ys_vv: ys_vv_test}),
        ref_logp_combined.eval(
            {
                x_vv: x_vv_test,
                y0_vv: ys_vv_test[0:1],
                y1_vv: ys_vv_test[1:2],
                y2_vv: ys_vv_test[2:3],
            }
        ),
    )


@pytest.mark.parametrize("stack", [False, True])
@pytest.mark.parametrize("shift", [0, 1])
def test_stack_split_component(stack, shift):
    x, y = pt.split(pt.random.normal(size=2), [1, 1], n_splits=2)
    component = x.squeeze() if stack else x
    if shift:
        component = component + shift
    dependent = pt.random.normal(component + 3)
    joined = pt.stack([component, dependent]) if stack else pt.concatenate([component, dependent])
    joined_value, y_value = joined.type(), y.type()

    joined_logp, y_logp = conditional_logp({joined: joined_value, y: y_value}).values()
    assert_no_rvs(joined_logp)
    assert_no_rvs(y_logp)
    fn = pytensor.function([joined_value, y_value], [joined_logp, y_logp])

    joined_test = np.array([1.0, 2.0])
    y_test = np.array([3.0])
    joined_result, y_result = fn(joined_test, y_test)
    np.testing.assert_allclose(
        joined_result,
        [
            st.norm.logpdf(joined_test[0] - shift),
            st.norm.logpdf(joined_test[1], joined_test[0] + 3),
        ],
    )
    np.testing.assert_allclose(y_result, st.norm.logpdf(y_test))


def test_nested_join_conditions_split_components():
    mu = pt.random.normal(size=1)
    x, y = pt.split(pt.random.normal(mu, size=4), [2, 2], n_splits=2)
    joined = pt.concatenate([mu, pt.exp(pt.concatenate([x, y]))])
    value = joined.type()

    joined_logp = logp(joined, value)
    assert_no_rvs(joined_logp)
    fn = pytensor.function([value], joined_logp.sum())

    test_value = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    expected = (
        st.norm.logpdf(test_value[0])
        + st.norm.logpdf(np.log(test_value[1:]), test_value[0]).sum()
        - np.log(test_value[1:]).sum()
    )
    np.testing.assert_allclose(fn(test_value), expected)


def test_nested_split_queries_with_conditioning():
    mu = pt.random.normal(size=1)
    x, y = pt.split(pt.random.normal(mu, size=4), [2, 2], n_splits=2)
    x1, x2 = pt.split(x, [1, 1], n_splits=2)
    joined = pt.concatenate([mu, x1, x2, y])
    value = joined.type()

    joined_logp = logp(joined, value)
    assert_no_rvs(joined_logp)
    fn = pytensor.function([value], joined_logp.sum())

    test_value = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    expected = st.norm.logpdf(test_value[0]) + st.norm.logpdf(test_value[1:], test_value[0]).sum()
    np.testing.assert_allclose(fn(test_value), expected)


def test_measurable_join_with_constant_input():
    base1_rv = pt.random.normal(size=(2,), name="base1")
    base2_rv = pt.random.exponential(size=(3,), name="base2")
    const = pt.constant(np.array([0.0, 0.0, 0.0]))
    y_rv = pt.join(0, base1_rv, const, base2_rv)
    y_rv.name = "y"

    base1_vv = base1_rv.clone()
    base2_vv = base2_rv.clone()
    y_vv = y_rv.clone()

    ref_logp = conditional_logp({base1_rv: base1_vv, base2_rv: base2_vv})
    ref_logp_combined = pt.sum([pt.sum(factor) for factor in ref_logp.values()])
    y_logp = logp(y_rv, y_vv)

    base1_testval = base1_rv.eval()
    base2_testval = base2_rv.eval()
    y_testval = np.concatenate([base1_testval, np.zeros(3), base2_testval]).astype(y_vv.dtype)

    ref_logp_eval = ref_logp_combined.eval({base1_vv: base1_testval, base2_vv: base2_testval})
    y_logp_eval = y_logp.eval({y_vv: y_testval})

    assert y_logp_eval.shape == y_testval.shape
    assert np.isclose(y_logp_eval.sum(), ref_logp_eval)

    y_testval_bad = y_testval.copy()
    y_testval_bad[2] = 1.0
    y_logp_eval_bad = y_logp.eval({y_vv: y_testval_bad})
    assert y_logp_eval_bad[2] == -np.inf


@pytest.mark.parametrize(
    "size1, size2, axis, concatenate",
    [
        ((5,), (3,), 0, True),
        ((5,), (3,), -1, True),
        ((5, 2), (3, 2), 0, True),
        ((2, 5), (2, 3), 1, True),
        ((2, 5), (2, 5), 0, False),
        ((2, 5), (2, 5), 1, False),
        ((2, 5), (2, 5), 2, False),
    ],
)
def test_measurable_join_univariate(size1, size2, axis, concatenate):
    base1_rv = pt.random.normal(size=size1, name="base1")
    base2_rv = pt.random.exponential(size=size2, name="base2")
    if concatenate:
        y_rv = pt.concatenate((base1_rv, base2_rv), axis=axis)
    else:
        y_rv = pt.stack((base1_rv, base2_rv), axis=axis)
    y_rv.name = "y"

    base1_vv = base1_rv.clone()
    base2_vv = base2_rv.clone()
    y_vv = y_rv.clone()

    base_logps = list(conditional_logp({base1_rv: base1_vv, base2_rv: base2_vv}).values())
    if concatenate:
        base_logps = pt.concatenate(base_logps, axis=axis)
    else:
        base_logps = pt.stack(base_logps, axis=axis)
    y_logp = logp(y_rv, y_vv)
    assert_no_rvs(y_logp)

    base1_testval = base1_rv.eval()
    base2_testval = base2_rv.eval()
    if concatenate:
        y_testval = np.concatenate((base1_testval, base2_testval), axis=axis)
    else:
        y_testval = np.stack((base1_testval, base2_testval), axis=axis)
    np.testing.assert_allclose(
        base_logps.eval({base1_vv: base1_testval, base2_vv: base2_testval}),
        y_logp.eval({y_vv: y_testval}),
    )


@pytest.mark.parametrize(
    "size1, supp_size1, size2, supp_size2, axis, concatenate, logp_axis",
    [
        (None, 2, None, 2, 0, True, 0),
        (None, 2, None, 2, -1, True, 0),
        ((5,), 2, (3,), 2, 0, True, 0),
        ((5,), 2, (3,), 2, -2, True, 0),
        ((2,), 5, (2,), 3, 1, True, 0),
        ((5, 6), 10, (5, 1), 10, 1, True, 1),
        ((5, 6), 10, (5, 1), 10, -2, True, 1),
        ((2,), 5, (2,), 5, 0, False, 0),
        ((2,), 5, (2,), 5, 1, False, 1),
        ((5, 6), 10, (5, 6), 10, 2, False, 2),
    ],
)
def test_measurable_join_multivariate(
    size1, supp_size1, size2, supp_size2, axis, concatenate, logp_axis
):
    base1_rv = pt.random.multivariate_normal(
        np.zeros(supp_size1), np.eye(supp_size1), size=size1, name="base1"
    )
    base2_rv = pt.random.dirichlet(np.ones(supp_size2), size=size2, name="base2")
    if concatenate:
        y_rv = pt.concatenate((base1_rv, base2_rv), axis=axis)
    else:
        y_rv = pt.stack((base1_rv, base2_rv), axis=axis)
    y_rv.name = "y"

    base1_vv = base1_rv.clone()
    base2_vv = base2_rv.clone()
    y_vv = y_rv.clone()

    y_logp = logp(y_rv, y_vv)
    assert_no_rvs(y_logp)

    base_logps = [
        pt.atleast_1d(logp)
        for logp in conditional_logp({base1_rv: base1_vv, base2_rv: base2_vv}).values()
    ]
    if concatenate:
        expected_logp = pt.concatenate(base_logps, axis=logp_axis)
    else:
        expected_logp = pt.stack(base_logps, axis=logp_axis)

    base1_testval = base1_rv.eval()
    base2_testval = base2_rv.eval()
    if concatenate:
        y_testval = np.concatenate((base1_testval, base2_testval), axis=axis)
    else:
        y_testval = np.stack((base1_testval, base2_testval), axis=axis)
    np.testing.assert_allclose(
        expected_logp.eval({base1_vv: base1_testval, base2_vv: base2_testval}),
        y_logp.eval({y_vv: y_testval}),
    )


def test_join_mixed_ndim_supp():
    base1_rv = pt.random.normal(size=3, name="base1")
    base2_rv = pt.random.dirichlet(np.ones(3), name="base2")
    y_rv = pt.concatenate((base1_rv, base2_rv), axis=0)

    y_vv = y_rv.clone()
    with pytest.raises(ValueError, match="Joined logps have different number of dimensions"):
        logp(y_rv, y_vv)


@pytensor.config.change_flags(cxx="")
@pytest.mark.parametrize(
    "ds_order",
    [
        (0, 2, 1),  # Swap
        (2, 1, 0),  # Swap
        (1, 2, 0),  # Swap
        (0, 1, 2, "x"),  # Expand
        ("x", 0, 1, 2),  # Expand
        (0, 2),  # Drop
        (2, 0),  # Swap and drop
        (2, 1, "x", 0),  # Swap and expand
        ("x", 0, 2),  # Expand and drop
        (2, "x", 0),  # Swap, expand and drop
    ],
)
@pytest.mark.parametrize("multivariate", (False, True))
def test_measurable_dimshuffle(ds_order, multivariate):
    if multivariate:
        base_rv = pt.random.dirichlet([1, 2, 3], size=(2, 1))
    else:
        base_rv = pt.exp(pt.random.beta(1, 2, size=(2, 1, 3)))

    ds_rv = base_rv.dimshuffle(ds_order)
    base_vv = base_rv.clone()
    ds_vv = ds_rv.clone()

    # Remove support dimension axis from ds_order (i.e., 2, for multivariate)
    if multivariate:
        logp_ds_order = [o for o in ds_order if o == "x" or o < 2]
    else:
        logp_ds_order = ds_order

    ref_logp = logp(base_rv, base_vv).dimshuffle(logp_ds_order)

    # Disable local_dimshuffle_rv_lift to test fallback logprob rewrite
    ir_rewriter = logprob_rewrites_db.query(
        RewriteDatabaseQuery(include=["basic"]).excluding("dimshuffle_lift")
    )
    ds_logp = conditional_logp({ds_rv: ds_vv}, ir_rewriter=ir_rewriter)
    ds_logp_combined = pt.add(*ds_logp.values())
    assert ds_logp_combined is not None

    ref_logp_fn = pytensor.function([base_vv], ref_logp)
    ds_logp_fn = pytensor.function([ds_vv], ds_logp_combined)

    base_test_value = base_rv.eval()
    ds_test_value = pt.constant(base_test_value).dimshuffle(ds_order).eval()

    np.testing.assert_array_equal(ref_logp_fn(base_test_value), ds_logp_fn(ds_test_value))


def test_cumsum_between_support_axis_dimshuffles():
    x = pt.random.dirichlet(np.ones(3), size=(4, 2))
    y = pt.cumsum(x.dimshuffle(0, 2, 1), axis=1).dimshuffle(1, 0, 2)
    value = y.type()
    term = conditional_logp({y: value})[value]
    assert_no_rvs(term)
    point = np.random.default_rng(42).dirichlet(np.ones(3), size=(4, 2))
    transformed_point = point.transpose(0, 2, 1).cumsum(axis=1).transpose(1, 0, 2)
    expected = st.dirichlet(np.ones(3)).logpdf(point.reshape(-1, 3).T).reshape(4, 2)
    np.testing.assert_allclose(term.eval({value: transformed_point}), expected)


class TestMeasurableSplit:
    def test_univariate(self):
        rng = np.random.default_rng(388)
        mu = np.arange(6)[:, None]
        sigma = np.arange(5) + 1

        x = pt.random.normal(mu, sigma, size=(6, 5), name="x")

        # axis=0
        x_parts = pt.split(x, splits_size=[2, 4], n_splits=2, axis=0)
        x_parts_vv = [x_part.clone() for x_part in x_parts]
        logp_parts = list(conditional_logp(dict(zip(x_parts, x_parts_vv))).values())

        logp_fn = pytensor.function(x_parts_vv, logp_parts)
        x_parts_test = [rng.normal(size=x_part.type.shape) for x_part in x_parts_vv]
        logp_x1_eval, logp_x2_eval = logp_fn(*x_parts_test)
        np.testing.assert_allclose(
            logp_x1_eval,
            st.norm.logpdf(x_parts_test[0], mu[:2], sigma),
        )
        np.testing.assert_allclose(
            logp_x2_eval,
            st.norm.logpdf(x_parts_test[1], mu[2:], sigma),
        )

        # axis=1
        x_parts = pt.split(x, splits_size=[2, 1, 2], n_splits=3, axis=1)
        x_parts_vv = [x_part.clone() for x_part in x_parts]
        logp_parts = list(conditional_logp(dict(zip(x_parts, x_parts_vv))).values())

        logp_fn = pytensor.function(x_parts_vv, logp_parts)
        x_parts_test = [rng.normal(size=x_part.type.shape) for x_part in x_parts_vv]
        logp_x1_eval, logp_x2_eval, logp_x3_eval = logp_fn(*x_parts_test)
        np.testing.assert_allclose(
            logp_x1_eval,
            st.norm.logpdf(x_parts_test[0], mu, sigma[:2]),
        )
        np.testing.assert_allclose(
            logp_x2_eval,
            st.norm.logpdf(x_parts_test[1], mu, sigma[2:3]),
        )
        np.testing.assert_allclose(
            logp_x3_eval,
            st.norm.logpdf(x_parts_test[2], mu, sigma[3:]),
        )

    def test_multivariate(self):
        @np.vectorize(signature=("(n),(n)->()"))
        def scipy_dirichlet_logpdf(x, alpha):
            """Compute the logpdf of a Dirichlet distribution using scipy."""
            return st.dirichlet.logpdf(x, alpha)

        # (3, 5) Dirichlet
        rng = np.random.default_rng(426)
        rng_pt = random_generator_type("rng")
        alpha = np.linspace(1, 10, 5) * np.array([1, 10, 100])[:, None]
        x = pt.random.dirichlet(alpha, rng=rng_pt)

        # axis=-2 (i.e., 0, - batch dimension)
        x_parts = pt.split(x, splits_size=[2, 1], n_splits=2, axis=-2)
        x_parts_vv = [x_part.clone() for x_part in x_parts]
        logp_parts = list(conditional_logp(dict(zip(x_parts, x_parts_vv))).values())
        assert logp_parts[0].type.shape == (2,)
        assert logp_parts[1].type.shape == (1,)

        logp_fn = pytensor.function(x_parts_vv, logp_parts)
        x_parts_test = pytensor.function([rng_pt], x_parts)(rng)
        logp_x1_eval, logp_x2_eval = logp_fn(*x_parts_test)
        np.testing.assert_allclose(
            logp_x1_eval,
            scipy_dirichlet_logpdf(x_parts_test[0], alpha[:2]),
        )
        np.testing.assert_allclose(
            logp_x2_eval,
            scipy_dirichlet_logpdf(x_parts_test[1], alpha[2:]),
        )

        # axis=-1 (i.e., 1, - support dimension)
        x_parts = pt.split(x, splits_size=[2, 3], n_splits=2, axis=-1)
        x_parts_vv = [x_part.clone() for x_part in x_parts]
        logp_parts = list(conditional_logp(dict(zip(x_parts, x_parts_vv))).values())

        logp_fn = pytensor.function(x_parts_vv, logp_parts)

        x_parts_test = pytensor.function([rng_pt], x_parts)(rng)
        logp_x1_eval, logp_x2_eval = logp_fn(*x_parts_test)
        np.testing.assert_allclose(logp_x1_eval * 3, logp_x2_eval * 2)
        logp_total = logp_x1_eval + logp_x2_eval
        np.testing.assert_allclose(
            logp_total,
            scipy_dirichlet_logpdf(np.concatenate(x_parts_test, axis=1), alpha),
        )

    def test_values_reached_through_measurable_chains(self):
        # Split logp must receive all values together to reconstruct the base variable.
        rng = np.random.default_rng(521)
        mu = np.arange(6)
        x = pt.random.normal(mu, 1.0, name="x")
        x_parts = pt.split(x, splits_size=[2, 4], n_splits=2, axis=0)
        x_parts_vv = [x_part.clone() for x_part in x_parts]
        x_parts_test = [rng.normal(size=x_part.type.shape) for x_part in x_parts_vv]

        # One part valued behind a shift, the other directly on the split node
        logp_parts = list(
            conditional_logp({x_parts[0] + 1: x_parts_vv[0], x_parts[1]: x_parts_vv[1]}).values()
        )
        logp_fn = pytensor.function(x_parts_vv, logp_parts)
        logp_x1_eval, logp_x2_eval = logp_fn(*x_parts_test)
        np.testing.assert_allclose(
            logp_x1_eval,
            st.norm.logpdf(x_parts_test[0] - 1, mu[:2]),
        )
        np.testing.assert_allclose(
            logp_x2_eval,
            st.norm.logpdf(x_parts_test[1], mu[2:]),
        )

        # Both parts valued behind a chain, so no value is attached to the split node itself
        logp_parts = list(
            conditional_logp(
                {x_parts[0] + 1: x_parts_vv[0], x_parts[1] * 2: x_parts_vv[1]}
            ).values()
        )
        logp_fn = pytensor.function(x_parts_vv, logp_parts)
        logp_x1_eval, logp_x2_eval = logp_fn(*x_parts_test)
        np.testing.assert_allclose(
            logp_x1_eval,
            st.norm.logpdf(x_parts_test[0] - 1, mu[:2]),
        )
        np.testing.assert_allclose(
            logp_x2_eval,
            st.norm.logpdf(x_parts_test[1] / 2, mu[2:]) - np.log(2),
        )

    @pytest.mark.xfail(
        reason="Rewrite from partial split to split on subtensor not implemented yet"
    )
    def test_not_all_splits_used(self):
        x = pt.random.normal(mu=pt.arange(6), name="x")
        x_parts = pt.split(x, splits_size=[2, 2, 2], n_splits=3, axis=0)[
            ::2
        ]  # Only use first two splits
        x_parts_vv = [x_part.clone() for x_part in x_parts]
        logp_parts = list(conditional_logp(dict(zip(x_parts, x_parts_vv))).values())
        assert len(logp_parts) == 2

        logp_fn = pytensor.function(x_parts_vv, logp_parts)
        x_parts_test = [x_part.eval() for x_part in x_parts_vv]
        logp_x1_eval, logp_x2_eval = logp_fn(*x_parts_test)
        np.testing.assert_allclose(
            logp_x1_eval,
            st.norm.logpdf(x_parts_test[0], loc=[0, 1]),
        )
        np.testing.assert_allclose(
            logp_x2_eval,
            st.norm.logpdf(x_parts_test[1], loc=[4, 5]),
        )

    def test_not_all_splits_used_core_dim(self):
        # Missing parts of a joint event require marginalization.
        x = pt.random.dirichlet(alphas=pt.ones(6), name="x")
        x_parts = pt.split(x, splits_size=[2, 2, 2], n_splits=3, axis=0)[
            :2
        ]  # Only use first two splits
        x_parts_vv = [x_part.clone() for x_part in x_parts]

        with pytest.raises(NotImplementedError, match="Partial Split"):
            conditional_logp(dict(zip(x_parts, x_parts_vv)))

    @pytest.mark.xfail(reason="Rewrite from subtensor to split not implemented yet")
    def test_subtensor_converted_to_splits(self):
        rng = np.random.default_rng(388)
        x = pt.random.normal(mu=pt.arange(5), name="x")

        x_parts = [x[:2], x[2:3], x[3:]]
        x_parts_vv = [x_part.clone() for x_part in x_parts]
        logp_parts = list(conditional_logp(dict(zip(x_parts, x_parts_vv))).values())
        assert len(logp_parts) == 3
        logp_fn = pytensor.function(x_parts_vv, logp_parts)
        x_parts_test = [rng.normal(size=x_part.type.shape) for x_part in x_parts_vv]
        logp_x1_eval, logp_x2_eval, logp_x3_eval = logp_fn(*x_parts_test)
        np.testing.assert_allclose(logp_x1_eval, st.norm.logpdf(x_parts_test[0], loc=[0, 1]))
        np.testing.assert_allclose(logp_x2_eval, st.norm.logpdf(x_parts_test[1], loc=[2]))
        np.testing.assert_allclose(logp_x3_eval, st.norm.logpdf(x_parts_test[2], loc=[3, 4]))


def test_measurable_broadcast():
    b_shape = pt.vector("b_shape", shape=(3,), dtype=int)

    x = pt.random.normal(size=(3, 1))
    bcast_x = pt.broadcast_to(x, shape=b_shape)
    bcast_x.name = "bcast_x"

    bcast_x_value = bcast_x.clone()
    logp_bcast_x = logp(bcast_x, bcast_x_value)
    logp_fn = pytensor.function([b_shape, bcast_x_value], logp_bcast_x, on_unused_input="ignore")

    # The expanded and broadcast dimensions are consumed like support dimensions:
    # the logp has the base variable's remaining batch shape
    # (assert_allclose also asserts shapes match, if neither is scalar)
    np.testing.assert_allclose(
        logp_fn([1, 3, 1], np.zeros((1, 3, 1))),
        st.norm.logpdf(np.zeros(3)),
    )
    np.testing.assert_allclose(
        logp_fn([1, 3, 5], np.zeros((1, 3, 5))),
        st.norm.logpdf(np.zeros(3)),
    )
    np.testing.assert_allclose(
        logp_fn([2, 3, 5], np.broadcast_to(np.arange(3).reshape(1, 3, 1), (2, 3, 5))),
        st.norm.logpdf(np.arange(3)),
    )
    # Invalid broadcast value
    np.testing.assert_array_equal(
        logp_fn([1, 3, 5], np.arange(3 * 5).reshape(1, 3, 5)),
        np.full(shape=(3,), fill_value=-np.inf),
    )
    # The invalidity check is elementwise over the base batch dimensions: an
    # inconsistent row only invalidates its own logp
    partially_valid = np.broadcast_to(np.arange(3).reshape(1, 3, 1), (1, 3, 5)).copy()
    partially_valid[0, 1, 3] = 99.0
    np.testing.assert_allclose(
        logp_fn([1, 3, 5], partially_valid),
        np.where([True, False, True], st.norm.logpdf(np.arange(3)), -np.inf),
    )


def test_measurable_broadcast_multivariate():
    x = pt.random.dirichlet(pt.ones(3), size=(1,))
    bcast_x = pt.broadcast_to(x, (5, 3))

    bcast_x_value = bcast_x.clone()
    logp_bcast_x = logp(bcast_x, bcast_x_value)

    rng = np.random.default_rng(170)
    row = rng.dirichlet(np.ones(3))
    valid_value = np.broadcast_to(row, (5, 3))
    valid_logp = logp_bcast_x.eval({bcast_x_value: valid_value})
    assert valid_logp.shape == ()
    np.testing.assert_allclose(
        valid_logp,
        st.dirichlet(np.ones(3)).logpdf(row),
    )

    invalid_value = rng.dirichlet(np.ones(3), size=(5,))
    np.testing.assert_array_equal(
        logp_bcast_x.eval({bcast_x_value: invalid_value}),
        -np.inf,
    )


def test_broadcast_not_measurable_behind_other_ops():
    # The broadcast dimensions are degenerate copies; other rewrites would treat them
    # as independent entries (e.g., counting the jacobian of the exp once per copy),
    # so the broadcast is only measurable when directly valued
    x = pt.random.normal()
    y = pt.exp(pt.broadcast_to(x, (3,)))
    with pytest.raises(NotImplementedError):
        logp(y, y.clone())


class TestMeasurableCast:
    def test_float_to_float(self):
        y = pt.cast(pt.random.normal(0.5, 1), "float32")
        y_vv = y.clone()

        y_test = np.float32(0.7)
        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: y_test}),
            st.norm(0.5, 1).logpdf(y_test),
        )
        np.testing.assert_allclose(
            logcdf(y, y_vv).eval({y_vv: y_test}),
            st.norm(0.5, 1).logcdf(y_test),
        )
        np.testing.assert_allclose(
            icdf(y, 0.3).eval(),
            st.norm(0.5, 1).ppf(0.3),
        )

    def test_discrete_to_float(self):
        y = pt.cast(pt.random.poisson(3), "float64")
        y_vv = y.clone()

        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: 3.0}),
            st.poisson(3).logpmf(3),
        )
        # P(cast(X) <= 3.9) = P(X <= 3)
        np.testing.assert_allclose(
            logcdf(y, y_vv).eval({y_vv: 3.9}),
            st.poisson(3).logcdf(3),
        )

        bern = pt.cast(pt.random.bernoulli(0.3), "float64")
        bern_icdf = icdf(bern, 0.8)
        assert bern_icdf.type.dtype == "float64"
        np.testing.assert_allclose(bern_icdf.eval(), st.bernoulli(0.3).ppf(0.8))

    def test_bool_to_int(self):
        y = pt.cast(pt.random.bernoulli(0.3), "int64")
        y_vv = y.clone()
        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: 1}),
            st.bernoulli(0.3).logpmf(1),
        )

    @pytest.mark.parametrize(
        "value, lower, upper",
        [(1, 1, 2), (0, -1, 1), (-1, -2, -1)],
        ids=["positive", "zero", "negative"],
    )
    def test_float_to_int(self, value, lower, upper):
        # The cast rounds towards zero, pooling (-1, 1) at zero
        y = pt.cast(pt.random.normal(0.5, 1), "int64")
        y_vv = y.clone()

        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: value}),
            np.log(st.norm(0.5, 1).cdf(upper) - st.norm(0.5, 1).cdf(lower)),
        )

    @pytest.mark.parametrize("rounding_fn", [pt.trunc, pt.floor, pt.ceil, pt.round])
    def test_rounded_float_to_int(self, rounding_fn):
        # The base variable is already supported on the integers, so the cast only
        # relabels the dtype and must not introduce a second truncation
        x = pt.random.normal(0.5, 1)
        y = pt.cast(rounding_fn(x), "int64")
        y_vv = y.clone()

        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: 1}),
            logp(rounding_fn(x), pt.constant(1.0)).eval(),
        )

    @pytest.mark.parametrize("out_dtype", ["bool", "uint8"])
    def test_non_truncating_discretizing_cast_not_measurable(self, out_dtype):
        # Casting to bool tests `x != 0` and casting to an unsigned int wraps around
        # for negative values; neither is the truncation that float -> int performs
        y = pt.cast(pt.random.normal(), out_dtype)
        with pytest.raises(NotImplementedError):
            logp(y, y.clone())

    def test_indirect_discrete_to_float_not_measurable(self):
        # If the cast is not directly valued, downstream rewrites would classify the
        # discrete base variable as continuous (e.g., applying a continuous jacobian)
        y = pt.exp(pt.cast(pt.random.poisson(3), "float64"))
        with pytest.raises(NotImplementedError):
            logp(y, y.clone())


class TestMeasurableIdentityOps:
    def test_scalar_from_tensor(self):
        y = pt.scalar_from_tensor(pt.random.normal(0.5, 1))
        y_vv = y.clone()
        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: 0.7}),
            st.norm(0.5, 1).logpdf(0.7),
        )
        np.testing.assert_allclose(
            logcdf(y, y_vv).eval({y_vv: 0.7}),
            st.norm(0.5, 1).logcdf(0.7),
        )
        np.testing.assert_allclose(
            icdf(y, 0.3).eval(),
            st.norm(0.5, 1).ppf(0.3),
        )

    def test_specify_assumptions(self):
        y = assume(pt.random.normal(pt.arange(4), 1, size=(4,)), "unique_indices")
        y_vv = y.clone()
        y_test = np.zeros(4)
        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: y_test}),
            st.norm(np.arange(4), 1).logpdf(y_test),
        )

        # Identity ops keep composing with other measurable rewrites
        y_exp = pt.exp(assume(pt.random.normal(0.5, 1), "diagonal"))
        y_exp_vv = y_exp.clone()
        np.testing.assert_allclose(
            logp(y_exp, y_exp_vv).eval({y_exp_vv: 2.0}),
            st.lognorm(s=1, scale=np.exp(0.5)).logpdf(2.0),
        )

    def test_deep_copy(self):
        y = DeepCopyOp()(pt.random.normal(0.5, 1))
        y_vv = y.clone()
        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: 0.7}),
            st.norm(0.5, 1).logpdf(0.7),
        )


class TestMeasurableJoinSplitDims:
    def test_join_dims(self):
        rng = np.random.default_rng(163)
        x = pt.random.normal(pt.arange(6).reshape((2, 3)), 1, size=(2, 3))
        y = pt.join_dims(x)
        y_vv = y.clone()

        y_test = rng.normal(size=6)
        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: y_test}),
            st.norm(np.arange(6), 1).logpdf(y_test),
        )

    def test_split_dims(self):
        rng = np.random.default_rng(164)
        x = pt.random.normal(pt.arange(6), 1, size=(6,))
        y = pt.split_dims(x, shape=(2, 3), axis=0)
        y_vv = y.clone()

        y_test = rng.normal(size=(2, 3))
        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: y_test}),
            st.norm(np.arange(6).reshape((2, 3)), 1).logpdf(y_test),
        )

    @pytest.mark.parametrize("transform_first", (False, True))
    def test_elemwise_chain(self, transform_first):
        rng = np.random.default_rng(165)
        x = pt.random.normal(pt.arange(6).reshape((2, 3)), 1, size=(2, 3))
        y = pt.join_dims(pt.exp(x)) if transform_first else pt.exp(pt.join_dims(x))
        y_vv = y.clone()

        y_test = np.abs(rng.normal(size=6)) + 0.1
        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: y_test}),
            st.norm(np.arange(6), 1).logpdf(np.log(y_test)) - np.log(y_test),
        )

    def test_multivariate_directly_valued(self):
        rng = np.random.default_rng(166)

        # The joined region extends into the support dimension consumed by the logp,
        # so only the remaining batch dimension is re-joined
        x = pt.random.dirichlet(pt.ones(3), size=(2,))
        y = pt.join_dims(x)
        y_vv = y.clone()
        y_test = rng.dirichlet(np.ones(3), size=2).ravel()
        y_logp = logp(y, y_vv).eval({y_vv: y_test})
        assert y_logp.shape == (2,)
        np.testing.assert_allclose(
            y_logp,
            st.dirichlet(np.ones(3)).logpdf(y_test.reshape((2, 3)).T),
        )

        # The split dimension is the support dimension itself
        x2 = pt.random.dirichlet(pt.ones(6))
        y2 = pt.split_dims(x2, shape=(2, 3), axis=0)
        y2_vv = y2.clone()
        y2_test = rng.dirichlet(np.ones(6)).reshape((2, 3))
        y2_logp = logp(y2, y2_vv).eval({y2_vv: y2_test})
        assert y2_logp.shape == ()
        np.testing.assert_allclose(
            y2_logp,
            st.dirichlet(np.ones(6)).logpdf(y2_test.ravel()),
        )

    def test_multivariate_indirect_join_within_batch(self):
        # A join contained in the batch axes leaves the support axes rightmost,
        # so it is measurable even behind other operations
        rng = np.random.default_rng(168)
        x = pt.random.dirichlet(pt.ones(3), size=(2, 2))
        y = pt.exp(pt.join_dims(x, start_axis=0, n_axes=2))
        y_vv = y.clone()
        y_test = np.exp(rng.dirichlet(np.ones(3), size=(2, 2)).reshape((4, 3)))
        y_logp = logp(y, y_vv).eval({y_vv: y_test})
        assert y_logp.shape == (4,)
        np.testing.assert_allclose(
            y_logp,
            st.dirichlet(np.ones(3)).logpdf(np.log(y_test).T) - np.log(y_test).sum(-1),
        )

    def test_multivariate_indirect_join_within_support(self):
        # A join contained in the support axes just deflates them into fewer
        # rightmost axes, so it is measurable even behind other operations
        rng = np.random.default_rng(169)
        x = pm.MatrixNormal.dist(mu=np.zeros((2, 3)), rowcov=np.eye(2), colcov=np.eye(3))
        y = pt.exp(pt.join_dims(x, start_axis=0, n_axes=2))
        y_vv = y.clone()
        y_test = np.exp(rng.normal(size=6))
        # With identity covariances the matrix normal entries are iid standard normal
        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: y_test}),
            st.norm.logpdf(np.log(y_test)).sum() - np.log(y_test).sum(),
        )

    # batch size 1 is the treacherous case: if the straddling join were wrongly
    # claimed, the truncated logp would silently broadcast against the fused value
    # dimension instead of raising a shape error
    @pytest.mark.parametrize("batch_size", (1, 2))
    def test_multivariate_indirect_straddling_join_not_measurable(self, batch_size):
        # A join straddling the batch/support boundary merges batch and support axes;
        # it is only measurable when directly valued
        x = pt.random.dirichlet(pt.ones(3), size=(batch_size,))
        y = pt.exp(pt.join_dims(x))
        with pytest.raises(NotImplementedError):
            logp(y, y.clone())

    def test_multivariate_indirect_split(self):
        # Splitting only inflates a single axis, so it is measurable even for
        # multivariate variables behind other operations
        rng = np.random.default_rng(167)
        x = pt.random.dirichlet(pt.ones(6))
        y = pt.exp(pt.split_dims(x, shape=(2, 3), axis=0))
        y_vv = y.clone()
        y_test = np.exp(rng.dirichlet(np.ones(6)).reshape((2, 3)))
        np.testing.assert_allclose(
            logp(y, y_vv).eval({y_vv: y_test}),
            st.dirichlet(np.ones(6)).logpdf(np.log(y_test).ravel()) - np.log(y_test).sum(),
        )


def test_symbolic_split_join():
    lengths = pt.lvector("lengths", shape=(2,))
    base = pt.random.normal(size=lengths.sum())
    x, y = pt.split(base, lengths + 0, n_splits=2)
    u = pt.random.normal(x + 3)
    w = pt.concatenate([x, u])
    w_value, y_value = pt.vector("w_value"), pt.vector("y_value")
    terms = conditional_logp({w: w_value, y: y_value})
    for term in terms.values():
        assert_no_rvs(term)
    fn = pytensor.function([lengths, w_value, y_value], list(terms.values()))
    for sizes, wv, yv in [([1, 2], [1.0, 2.0], [3.0, 4.0]), ([2, 1], [1.0, 2.0, 4.0, 7.0], [3.0])]:
        n = sizes[0]
        actual_w, actual_y = fn(sizes, wv, yv)
        np.testing.assert_allclose(
            actual_w, np.r_[st.norm.logpdf(wv[:n]), st.norm.logpdf(wv[n:], np.asarray(wv[:n]) + 3)]
        )
        np.testing.assert_allclose(actual_y, st.norm.logpdf(yv))


def test_transposed_multivariate_transform():
    mean = pt.matrix("mean", shape=(None, 3))
    base = pm.MvNormal.dist(mu=mean, cov=np.eye(3))
    observed = pt.exp(base.T)
    value = pt.matrix("value")
    terms = conditional_logp({observed: value})
    assert_no_rvs(terms[value])
    fn = pytensor.function([mean, value], terms[value])
    for n in [2, 4]:
        point = np.exp(np.arange(n * 3).reshape((3, n)) / 10)
        mu = np.zeros((n, 3))
        expected = st.multivariate_normal.logpdf(np.log(point).T, cov=np.eye(3)) - np.log(
            point
        ).sum(axis=0)
        np.testing.assert_allclose(fn(mu, point), expected)


def test_concatenate_scalar_multivariate_densities():
    mean = pt.vector("mean")
    n = mean.shape[0]
    x = pm.MvNormal.dist(mu=mean, cov=pt.eye(n))
    y = pm.MvNormal.dist(mu=x + 2, cov=pt.eye(n))
    observed = pt.concatenate([x, y])
    value = pt.vector("value")
    terms = conditional_logp({observed: value})
    assert_no_rvs(terms[value])
    assert terms[value].ndim == 1
    fn = pytensor.function([mean, value], terms[value])
    for size in [2, 3]:
        x_value = np.arange(size) / 10
        y_value = x_value + 3
        actual = fn(np.zeros(size), np.r_[x_value, y_value])
        expected = [
            st.multivariate_normal.logpdf(x_value, cov=np.eye(size)),
            st.multivariate_normal.logpdf(y_value, mean=x_value + 2, cov=np.eye(size)),
        ]
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_density_dtype_contract(dtype):
    with pytensor.config.change_flags(floatX=dtype):
        x = pt.random.normal(size=2)
        observed = pt.exp(x)
        value = observed.type()
        terms = conditional_logp({observed: value})
        term = terms[value]
        assert term.dtype == dtype
        actual = pytensor.function([value], term)(np.array([1.0, 2.0], dtype=dtype))
        np.testing.assert_allclose(
            actual, st.norm.logpdf(np.log([1.0, 2.0])) - np.log([1.0, 2.0]), rtol=1e-6
        )
