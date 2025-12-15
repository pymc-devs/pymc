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
import pytensor.tensor as pt
import pytest
import scipy as sp
import scipy.stats as st

from pymc import logp
from pymc.logprob import conditional_logp
from pymc.logprob.transform_value import TransformValuesRewrite
from pymc.logprob.transforms import LogTransform
from pymc.testing import assert_no_rvs


def test_continuous_rv_clip():
    x_rv = pt.random.normal(0.5, 1)
    cens_x_rv = pt.clip(x_rv, -2, 2)

    cens_x_vv = cens_x_rv.clone()
    cens_x_vv.tag.test_value = 0

    logprob = pt.sum(logp(cens_x_rv, cens_x_vv))
    assert_no_rvs(logprob)

    logp_fn = pytensor.function([cens_x_vv], logprob)
    ref_scipy = st.norm(0.5, 1)

    assert logp_fn(-3) == -np.inf
    assert logp_fn(3) == -np.inf

    assert np.isclose(logp_fn(-2), ref_scipy.logcdf(-2))
    assert np.isclose(logp_fn(2), ref_scipy.logsf(2))
    assert np.isclose(logp_fn(0), ref_scipy.logpdf(0))


def test_discrete_rv_clip():
    x_rv = pt.random.poisson(2)
    cens_x_rv = pt.clip(x_rv, 1, 4)

    cens_x_vv = cens_x_rv.clone()

    logprob = pt.sum(logp(cens_x_rv, cens_x_vv))
    assert_no_rvs(logprob)

    logp_fn = pytensor.function([cens_x_vv], logprob)
    ref_scipy = st.poisson(2)

    assert logp_fn(0) == -np.inf
    assert logp_fn(5) == -np.inf

    assert np.isclose(logp_fn(1), ref_scipy.logcdf(1))
    assert np.isclose(logp_fn(4), np.logaddexp(ref_scipy.logsf(4), ref_scipy.logpmf(4)))
    assert np.isclose(logp_fn(2), ref_scipy.logpmf(2))


def test_one_sided_clip():
    x_rv = pt.random.normal(0, 1)
    lb_cens_x_rv = pt.clip(x_rv, -1, x_rv)
    ub_cens_x_rv = pt.clip(x_rv, x_rv, 1)

    lb_cens_x_vv = lb_cens_x_rv.clone()
    ub_cens_x_vv = ub_cens_x_rv.clone()

    lb_logp = pt.sum(logp(lb_cens_x_rv, lb_cens_x_vv))
    ub_logp = pt.sum(logp(ub_cens_x_rv, ub_cens_x_vv))
    assert_no_rvs(lb_logp)
    assert_no_rvs(ub_logp)

    logp_fn = pytensor.function([lb_cens_x_vv, ub_cens_x_vv], [lb_logp, ub_logp])
    ref_scipy = st.norm(0, 1)

    assert np.all(np.array(logp_fn(-2, 2)) == -np.inf)
    assert np.all(np.array(logp_fn(2, -2)) != -np.inf)
    np.testing.assert_almost_equal(logp_fn(-1, 1), ref_scipy.logcdf(-1))
    np.testing.assert_almost_equal(logp_fn(1, -1), ref_scipy.logpdf(-1))


def test_useless_clip():
    x_rv = pt.random.normal(0.5, 1, size=3)
    cens_x_rv = pt.clip(x_rv, x_rv, x_rv)

    cens_x_vv = cens_x_rv.clone()

    logprob = logp(cens_x_rv, cens_x_vv)
    assert_no_rvs(logprob)

    logp_fn = pytensor.function([cens_x_vv], logprob)
    ref_scipy = st.norm(0.5, 1)

    np.testing.assert_allclose(logp_fn([-2, 0, 2]), ref_scipy.logpdf([-2, 0, 2]))


def test_random_clip():
    lb_rv = pt.random.normal(0, 1, size=2)
    x_rv = pt.random.normal(0, 2)
    cens_x_rv = pt.clip(x_rv, lb_rv, [1, 1])

    lb_vv = lb_rv.clone()
    cens_x_vv = cens_x_rv.clone()
    logp = conditional_logp({cens_x_rv: cens_x_vv, lb_rv: lb_vv})
    logp_combined = pt.add(*logp.values())

    assert_no_rvs(logp_combined)

    logp_fn = pytensor.function([lb_vv, cens_x_vv], logp_combined)
    res = logp_fn([0, -1], [-1, -1])
    assert res[0] == -np.inf
    assert res[1] != -np.inf


def test_broadcasted_clip_constant():
    lb_rv = pt.random.uniform(0, 1)
    x_rv = pt.random.normal(0, 2)
    cens_x_rv = pt.clip(x_rv, lb_rv, [1, 1])

    lb_vv = lb_rv.clone()
    cens_x_vv = cens_x_rv.clone()

    logp = conditional_logp({cens_x_rv: cens_x_vv, lb_rv: lb_vv})
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

    assert_no_rvs(logp_combined)


def test_broadcasted_clip_random():
    lb_rv = pt.random.normal(0, 1)
    x_rv = pt.random.normal(0, 2, size=2)
    cens_x_rv = pt.clip(x_rv, lb_rv, 1)

    lb_vv = lb_rv.clone()
    cens_x_vv = cens_x_rv.clone()

    logp = conditional_logp({cens_x_rv: cens_x_vv, lb_rv: lb_vv})
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

    assert_no_rvs(logp_combined)


def test_fail_base_and_clip_have_values():
    """Test failure when both base_rv and clipped_rv are given value vars"""
    x_rv = pt.random.normal(0, 1)
    cens_x_rv = pt.clip(x_rv, x_rv, 1)
    cens_x_rv.name = "cens_x"

    x_vv = x_rv.clone()
    cens_x_vv = cens_x_rv.clone()
    with pytest.raises(RuntimeError, match="could not be derived: {cens_x}"):
        conditional_logp({cens_x_rv: cens_x_vv, x_rv: x_vv})


def test_fail_multiple_clip_single_base():
    """Test failure when multiple clipped_rvs share a single base_rv"""
    base_rv = pt.random.normal(0, 1)
    cens_rv1 = pt.clip(base_rv, -1, 1)
    cens_rv1.name = "cens1"
    cens_rv2 = pt.clip(base_rv, -1, 1)
    cens_rv2.name = "cens2"

    cens_vv1 = cens_rv1.clone()
    cens_vv2 = cens_rv2.clone()
    with pytest.raises(ValueError, match="too many values to unpack"):
        conditional_logp({cens_rv1: cens_vv1, cens_rv2: cens_vv2})


def test_deterministic_clipping():
    x_rv = pt.random.normal(0, 1)
    clip = pt.clip(x_rv, 0, 0)
    y_rv = pt.random.normal(clip, 1)

    x_vv = x_rv.clone()
    y_vv = y_rv.clone()
    logp = conditional_logp({x_rv: x_vv, y_rv: y_vv})
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])
    assert_no_rvs(logp_combined)

    logp_fn = pytensor.function([x_vv, y_vv], logp_combined)
    assert np.isclose(
        logp_fn(-1, 1),
        st.norm(0, 1).logpdf(-1) + st.norm(0, 1).logpdf(1),
    )


def test_clip_transform():
    x_rv = pt.random.normal(0.5, 1)
    cens_x_rv = pt.clip(x_rv, 0, x_rv)

    cens_x_vv = cens_x_rv.clone()

    transform = TransformValuesRewrite({cens_x_vv: LogTransform()})
    logp = conditional_logp({cens_x_rv: cens_x_vv}, extra_rewrites=transform)
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

    cens_x_vv_testval = -1
    obs_logp = logp_combined.eval({cens_x_vv: cens_x_vv_testval})
    exp_logp = sp.stats.norm(0.5, 1).logpdf(np.exp(cens_x_vv_testval)) + cens_x_vv_testval

    assert np.isclose(obs_logp, exp_logp)


@pytest.mark.parametrize("rounding_op", (pt.round, pt.floor, pt.ceil))
def test_rounding(rounding_op):
    loc = 1
    scale = 2
    test_value = np.arange(-3, 4)

    x = pt.random.normal(loc, scale, size=test_value.shape, name="x")
    xr = rounding_op(x)
    xr.name = "xr"

    xr_vv = xr.clone()
    logprob = logp(xr, xr_vv)
    assert logprob is not None

    x_sp = st.norm(loc, scale)
    if rounding_op == pt.round:
        expected_logp = np.log(x_sp.cdf(test_value + 0.5) - x_sp.cdf(test_value - 0.5))
    elif rounding_op == pt.floor:
        expected_logp = np.log(x_sp.cdf(test_value + 1.0) - x_sp.cdf(test_value))
    elif rounding_op == pt.ceil:
        expected_logp = np.log(x_sp.cdf(test_value) - x_sp.cdf(test_value - 1.0))
    else:
        raise NotImplementedError()

    assert np.allclose(
        logprob.eval({xr_vv: test_value}),
        expected_logp,
    )


@pytest.mark.parametrize(
    "censoring_side,bound_value",
    [
        ("right", 40.0),  # Far right tail: CDF ≈ 1, need stable log(1-CDF)
        ("left", -40.0),  # Far left tail: CDF ≈ 0, need stable log(CDF)
    ],
)
def test_censored_logprob_numerical_stability(censoring_side, bound_value):
    """Test that censored distributions use numerically stable log-probability computations.

    For right-censoring at the upper bound, log(1 - CDF) is computed. When CDF ≈ 1
    (far right tail), this requires a stable logccdf implementation.

    For left-censoring at the lower bound, log(CDF) is computed. When CDF ≈ 0
    (far left tail), this requires a stable logcdf implementation.

    This test uses pm.Censored which is the high-level API for censored distributions.
    """
    import pymc as pm

    ref_scipy = st.norm(0, 1)

    with pm.Model() as model:
        normal_dist = pm.Normal.dist(mu=0.0, sigma=1.0)
        if censoring_side == "right":
            pm.Censored("y", normal_dist, lower=None, upper=bound_value)
            expected_logp = ref_scipy.logsf(bound_value)  # log(1 - CDF)
        else:  # left
            pm.Censored("y", normal_dist, lower=bound_value, upper=None)
            expected_logp = ref_scipy.logcdf(bound_value)  # log(CDF)

    # Compile the logp function
    logp_fn = model.compile_logp()

    # Evaluate at the bound - this is where the log survival/cdf function is used
    logp_at_bound = logp_fn({"y": bound_value})

    # This should be finite and correct, not -inf
    assert np.isfinite(logp_at_bound), (
        f"logp at {censoring_side} bound should be finite, got {logp_at_bound}"
    )
    assert np.isclose(logp_at_bound, expected_logp, rtol=1e-6), (
        f"logp at {censoring_side} bound: got {logp_at_bound}, expected {expected_logp}"
    )
