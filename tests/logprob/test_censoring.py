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
from pymc.logprob.basic import icdf, logcdf
from pymc.logprob.censoring import MeasurableClip
from pymc.logprob.rewriting import construct_ir_fgraph
from pymc.logprob.transform_value import TransformValuesRewrite
from pymc.logprob.transforms import LogTransform
from pymc.testing import assert_no_rvs


def test_continuous_rv_clip():
    x_rv = pt.random.normal(0.5, 1)
    cens_x_rv = pt.clip(x_rv, -2, 2)

    cens_x_vv = cens_x_rv.clone()

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


@pytest.mark.parametrize(
    "rounding_op", (pt.round, pt.round_half_away_from_zero, pt.floor, pt.ceil, pt.trunc)
)
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
    if rounding_op in (pt.round, pt.round_half_away_from_zero):
        expected_logp = np.log(x_sp.cdf(test_value + 0.5) - x_sp.cdf(test_value - 0.5))
    elif rounding_op == pt.floor:
        expected_logp = np.log(x_sp.cdf(test_value + 1.0) - x_sp.cdf(test_value))
    elif rounding_op == pt.ceil:
        expected_logp = np.log(x_sp.cdf(test_value) - x_sp.cdf(test_value - 1.0))
    elif rounding_op == pt.trunc:
        expected_logp = np.log(
            x_sp.cdf(test_value + (test_value >= 0)) - x_sp.cdf(test_value - (test_value <= 0))
        )
    else:
        raise NotImplementedError()

    assert np.allclose(
        logprob.eval({xr_vv: test_value}),
        expected_logp,
    )


@pytest.mark.parametrize("rounding_op", (pt.floor, pt.ceil, pt.trunc))
def test_rounding_discrete_base(rounding_op):
    # A variable that already sits on the integers is only upcast by the rounding, so
    # its own logprob applies unchanged (`pt.round` rejects integer inputs outright)
    mu = 3
    test_value = np.arange(0, 4)

    x = pt.random.poisson(mu, size=test_value.shape, name="x")
    xr = rounding_op(x)
    xr_vv = xr.clone()

    np.testing.assert_allclose(
        logp(xr, xr_vv).eval({xr_vv: test_value}),
        st.poisson(mu).logpmf(test_value),
    )


@pytest.mark.parametrize(
    "outer_op", (pt.round, pt.round_half_away_from_zero, pt.floor, pt.ceil, pt.trunc)
)
@pytest.mark.parametrize("inner_op", (pt.floor, pt.ceil, pt.trunc))
def test_rounding_rounded_base(outer_op, inner_op):
    # The inner rounding leaves the variable on the integers, where the outer one is an
    # identity, so the logprob is that of the inner rounding alone
    loc = 1
    scale = 2
    test_value = np.arange(-3, 4, dtype="float64")

    x = pt.random.normal(loc, scale, size=test_value.shape, name="x")
    inner = inner_op(x)

    outer_vv = outer_op(inner).clone()
    inner_vv = inner.clone()

    np.testing.assert_allclose(
        logp(outer_op(inner), outer_vv).eval({outer_vv: test_value}),
        logp(inner, inner_vv).eval({inner_vv: test_value}),
    )


@pytest.mark.parametrize(
    "rounding_op", (pt.round, pt.round_half_away_from_zero, pt.floor, pt.ceil, pt.trunc)
)
def test_rounding_censored_base_not_measurable(rounding_op):
    # A clipped variable pools mass at its bounds, so it is not continuous, but its
    # float dtype does not say so and we do not infer it (see
    # https://github.com/pymc-devs/pymc/issues/6360). Fail rather than treat the base as
    # continuous, which pools the mass at the upper bound into the neighbouring cell and
    # leaves the bound itself with a probability of zero.
    x = pt.random.normal(1, 2, size=(7,), name="x")
    y = rounding_op(pt.clip(x, 0, 3))

    with pytest.raises(NotImplementedError):
        logp(y, y.clone())


@pytest.mark.parametrize("swap_args", (False, True))
def test_maximum_minimum_censoring(swap_args):
    x_rv = pt.random.normal(0.5, 1)
    if swap_args:
        lb_cens_x_rv = pt.maximum(-1.0, x_rv)
        ub_cens_x_rv = pt.minimum(1.0, x_rv)
    else:
        lb_cens_x_rv = pt.maximum(x_rv, -1.0)
        ub_cens_x_rv = pt.minimum(x_rv, 1.0)

    lb_cens_x_vv = lb_cens_x_rv.clone()
    ub_cens_x_vv = ub_cens_x_rv.clone()

    lb_logp = logp(lb_cens_x_rv, lb_cens_x_vv)
    ub_logp = logp(ub_cens_x_rv, ub_cens_x_vv)
    assert_no_rvs(lb_logp)
    assert_no_rvs(ub_logp)

    logp_fn = pytensor.function([lb_cens_x_vv, ub_cens_x_vv], [lb_logp, ub_logp])
    ref_scipy = st.norm(0.5, 1)

    np.testing.assert_allclose(logp_fn(-1, 1), [ref_scipy.logcdf(-1), ref_scipy.logsf(1)])
    np.testing.assert_allclose(logp_fn(0, 0), ref_scipy.logpdf(0))
    assert np.all(np.array(logp_fn(-2, 2)) == -np.inf)


@pytest.mark.parametrize("max_of_min", (False, True))
def test_two_sided_maximum_minimum_censoring(max_of_min):
    x_rv = pt.random.normal(0.5, 1)
    if max_of_min:
        cens_x_rv = pt.maximum(pt.minimum(x_rv, 1.5), 0.3)
    else:
        cens_x_rv = pt.minimum(pt.maximum(x_rv, 0.3), 1.5)

    cens_x_vv = cens_x_rv.clone()
    logprob = logp(cens_x_rv, cens_x_vv)
    assert_no_rvs(logprob)

    logp_fn = pytensor.function([cens_x_vv], logprob)
    ref_scipy = st.norm(0.5, 1)

    assert logp_fn(0.0) == -np.inf
    assert logp_fn(2.0) == -np.inf
    np.testing.assert_allclose(logp_fn(0.3), ref_scipy.logcdf(0.3))
    np.testing.assert_allclose(logp_fn(1.5), ref_scipy.logsf(1.5))
    np.testing.assert_allclose(logp_fn(1.0), ref_scipy.logpdf(1.0))


def test_discrete_maximum_minimum_censoring():
    x_rv = pt.random.poisson(2)
    lb_cens_x_rv = pt.maximum(x_rv, 1)
    ub_cens_x_rv = pt.minimum(x_rv, 4)

    lb_cens_x_vv = lb_cens_x_rv.clone()
    ub_cens_x_vv = ub_cens_x_rv.clone()
    lb_logp = logp(lb_cens_x_rv, lb_cens_x_vv)
    ub_logp = logp(ub_cens_x_rv, ub_cens_x_vv)
    assert_no_rvs(lb_logp)
    assert_no_rvs(ub_logp)

    logp_fn = pytensor.function([lb_cens_x_vv, ub_cens_x_vv], [lb_logp, ub_logp])
    ref_scipy = st.poisson(2)

    np.testing.assert_allclose(
        logp_fn(1, 4),
        [ref_scipy.logcdf(1), np.logaddexp(ref_scipy.logsf(4), ref_scipy.logpmf(4))],
    )
    np.testing.assert_allclose(logp_fn(2, 2), ref_scipy.logpmf(2))
    assert np.all(np.array(logp_fn(0, 5)) == -np.inf)

    # Two-sided matches the equivalent clip
    cens_x_rv = pt.maximum(pt.minimum(x_rv, 4), 1)
    cens_x_vv = cens_x_rv.clone()
    two_sided_logp_fn = pytensor.function([cens_x_vv], logp(cens_x_rv, cens_x_vv))
    np.testing.assert_allclose(two_sided_logp_fn(1), ref_scipy.logcdf(1))
    np.testing.assert_allclose(
        two_sided_logp_fn(4), np.logaddexp(ref_scipy.logsf(4), ref_scipy.logpmf(4))
    )
    np.testing.assert_allclose(two_sided_logp_fn(2), ref_scipy.logpmf(2))


def test_maximum_of_two_rvs_not_claimed_as_censoring():
    x_rv = pt.random.normal()
    y_rv = pt.random.normal()
    z_rv = pt.maximum(x_rv, y_rv)

    with pytest.raises(NotImplementedError):
        logp(z_rv, z_rv.clone())


def test_clip_logcdf_icdf():
    x_rv = pt.random.normal(0.5, 1)
    cens_x_rv = pt.clip(x_rv, 0.3, 1.5)
    cens_x_vv = cens_x_rv.clone()

    cens_logcdf = logcdf(cens_x_rv, cens_x_vv)
    cens_icdf = icdf(cens_x_rv, cens_x_vv)
    logcdf_fn = pytensor.function([cens_x_vv], cens_logcdf)
    icdf_fn = pytensor.function([cens_x_vv], cens_icdf)
    ref_scipy = st.norm(0.5, 1)

    assert logcdf_fn(0.1) == -np.inf
    np.testing.assert_allclose(logcdf_fn(0.3), ref_scipy.logcdf(0.3))
    np.testing.assert_allclose(logcdf_fn(1.0), ref_scipy.logcdf(1.0))
    assert logcdf_fn(1.5) == 0.0
    assert logcdf_fn(2.0) == 0.0

    # The point masses at the bounds absorb the tail quantiles
    np.testing.assert_allclose(icdf_fn(0.05), 0.3)
    np.testing.assert_allclose(icdf_fn(0.5), ref_scipy.ppf(0.5))
    np.testing.assert_allclose(icdf_fn(0.99), 1.5)


def test_nested_clip_fusion():
    x_rv = pt.random.normal(0.5, 1)
    # Bounds combine with maximum/minimum: equivalent to clip(x, 0, 1)
    cens_x_rv = pt.clip(pt.clip(x_rv, -1.0, 1.0), 0.0, 2.0)
    cens_x_vv = cens_x_rv.clone()

    fgraph = construct_ir_fgraph({cens_x_rv: cens_x_vv})
    assert sum(isinstance(node.op, MeasurableClip) for node in fgraph.toposort()) == 1

    logp_fn = pytensor.function([cens_x_vv], logp(cens_x_rv, cens_x_vv))
    ref_scipy = st.norm(0.5, 1)

    np.testing.assert_allclose(logp_fn(0.0), ref_scipy.logcdf(0.0))
    np.testing.assert_allclose(logp_fn(1.0), ref_scipy.logsf(1.0))
    np.testing.assert_allclose(logp_fn(0.5), ref_scipy.logpdf(0.5))
    assert logp_fn(1.5) == -np.inf
