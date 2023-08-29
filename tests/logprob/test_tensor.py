#   Copyright 2023 The PyMC Developers
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
from pytensor.graph import RewriteDatabaseQuery
from pytensor.graph.rewriting.basic import in2out
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor.basic import Alloc
from scipy import stats as st

from pymc.logprob.basic import conditional_logp, logp
from pymc.logprob.rewriting import logprob_rewrites_db
from pymc.logprob.tensor import naive_bcast_rv_lift
from pymc.testing import assert_no_rvs


def test_naive_bcast_rv_lift():
    r"""Make sure `naive_bcast_rv_lift` can handle useless scalar `Alloc`\s."""
    X_rv = pt.random.normal()
    Z_at = Alloc()(X_rv, *())

    # Make sure we're testing what we intend to test
    assert isinstance(Z_at.owner.op, Alloc)

    res = rewrite_graph(Z_at, custom_rewrite=in2out(naive_bcast_rv_lift), clone=False)
    assert res is X_rv


def test_naive_bcast_rv_lift_valued_var():
    r"""Check that `naive_bcast_rv_lift` won't touch valued variables"""

    x_rv = pt.random.normal(name="x")
    broadcasted_x_rv = pt.broadcast_to(x_rv, (2,))

    y_rv = pt.random.normal(broadcasted_x_rv, name="y")

    x_vv = x_rv.clone()
    y_vv = y_rv.clone()
    logp_map = conditional_logp({x_rv: x_vv, y_rv: y_vv})
    assert x_vv in logp_map
    assert y_vv in logp_map
    assert len(logp_map) == 2
    assert np.allclose(logp_map[x_vv].eval({x_vv: 0}), st.norm(0).logpdf(0))
    assert np.allclose(logp_map[y_vv].eval({x_vv: 0, y_vv: [0, 0]}), st.norm(0).logpdf([0, 0]))


@pytest.mark.xfail(RuntimeError, reason="logprob for broadcasted RVs not implemented")
def test_bcast_rv_logp():
    """Test that derived logp for broadcasted RV is correct"""

    x_rv = pt.random.normal(name="x")
    broadcasted_x_rv = pt.broadcast_to(x_rv, (2,))
    broadcasted_x_rv.name = "broadcasted_x"
    broadcasted_x_vv = broadcasted_x_rv.clone()

    logp = conditional_logp({broadcasted_x_rv: broadcasted_x_vv})
    logp_combined = pt.add(*logp.values())
    valid_logp = logp_combined.eval({broadcasted_x_vv: [0, 0]})

    assert valid_logp.shape == ()
    assert np.isclose(valid_logp, st.norm.logpdf(0))

    # It's not possible for broadcasted dimensions to have different values
    # This should either raise or return -inf
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
    "size1, supp_size1, size2, supp_size2, axis, concatenate",
    [
        (None, 2, None, 2, 0, True),
        (None, 2, None, 2, -1, True),
        ((5,), 2, (3,), 2, 0, True),
        ((5,), 2, (3,), 2, -2, True),
        ((2,), 5, (2,), 3, 1, True),
        pytest.param(
            (2,),
            5,
            (2,),
            5,
            0,
            False,
            marks=pytest.mark.xfail(reason="cannot measure dimshuffled multivariate RVs"),
        ),
        pytest.param(
            (2,),
            5,
            (2,),
            5,
            1,
            False,
            marks=pytest.mark.xfail(reason="cannot measure dimshuffled multivariate RVs"),
        ),
    ],
)
def test_measurable_join_multivariate(size1, supp_size1, size2, supp_size2, axis, concatenate):
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
    base_logps = [
        pt.atleast_1d(logp)
        for logp in conditional_logp({base1_rv: base1_vv, base2_rv: base2_vv}).values()
    ]

    if concatenate:
        axis_norm = np.core.numeric.normalize_axis_index(axis, base1_rv.ndim)
        base_logps = pt.concatenate(base_logps, axis=axis_norm - 1)
    else:
        axis_norm = np.core.numeric.normalize_axis_index(axis, base1_rv.ndim + 1)
        base_logps = pt.stack(base_logps, axis=axis_norm - 1)
    y_logp = y_logp = logp(y_rv, y_vv)
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
        (
            0,
            2,
        ),  # Drop
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

    # Disable local_dimshuffle_rv_lift to test fallback Aeppl rewrite
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


def test_unmeargeable_dimshuffles():
    # Test that graphs with DimShuffles that cannot be lifted/merged fail

    # Initial support axis is at axis=-1
    x = pt.random.dirichlet(
        np.ones((3,)),
        size=(4, 2),
    )
    # Support axis is now at axis=-2
    y = x.dimshuffle((0, 2, 1))
    # Downstream dimshuffle will not be lifted through cumsum. If it ever is,
    # we will need a different measurable Op example
    z = pt.cumsum(y, axis=-2)
    # Support axis is now at axis=-3
    w = z.dimshuffle((1, 0, 2))

    w_vv = w.clone()
    # TODO: Check that logp is correct if this type of graphs is ever supported
    with pytest.raises(RuntimeError, match="could not be derived"):
        conditional_logp({w: w_vv})
