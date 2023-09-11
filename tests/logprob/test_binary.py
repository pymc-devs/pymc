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
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats as st

from pytensor import function

from pymc import logp
from pymc.logprob import conditional_logp
from pymc.testing import assert_no_rvs


@pytest.mark.parametrize(
    "comparison_op, exp_logp_true, exp_logp_false, inputs",
    [
        ((pt.lt, pt.le), "logcdf", "logsf", (pt.random.normal(0, 1), 0.5)),
        ((pt.gt, pt.ge), "logsf", "logcdf", (pt.random.normal(0, 1), 0.5)),
        ((pt.lt, pt.le), "logsf", "logcdf", (0.5, pt.random.normal(0, 1))),
        ((pt.gt, pt.ge), "logcdf", "logsf", (0.5, pt.random.normal(0, 1))),
    ],
)
def test_continuous_rv_comparison_bitwise(comparison_op, exp_logp_true, exp_logp_false, inputs):
    for op in comparison_op:
        comp_x_rv = op(*inputs)

        comp_x_vv = comp_x_rv.clone()

        logprob = logp(comp_x_rv, comp_x_vv)
        assert_no_rvs(logprob)

        logp_fn = pytensor.function([comp_x_vv], logprob)
        ref_scipy = st.norm(0, 1)

        assert np.isclose(logp_fn(0), getattr(ref_scipy, exp_logp_false)(0.5))
        assert np.isclose(logp_fn(1), getattr(ref_scipy, exp_logp_true)(0.5))

        bitwise_rv = pt.bitwise_not(op(*inputs))
        bitwise_vv = bitwise_rv.clone()

        logprob_not = logp(bitwise_rv, bitwise_vv)
        assert_no_rvs(logprob_not)

        logp_fn_not = pytensor.function([bitwise_vv], logprob_not)

        assert np.isclose(logp_fn_not(0), getattr(ref_scipy, exp_logp_true)(0.5))
        assert np.isclose(logp_fn_not(1), getattr(ref_scipy, exp_logp_false)(0.5))


@pytest.mark.parametrize(
    "comparison_op, exp_logp_true, exp_logp_false, inputs",
    [
        (
            pt.lt,
            lambda x: st.poisson(2).logcdf(x - 1),
            lambda x: np.logaddexp(st.poisson(2).logsf(x), st.poisson(2).logpmf(x)),
            (pt.random.poisson(2), 3),
        ),
        (
            pt.ge,
            lambda x: np.logaddexp(st.poisson(2).logsf(x), st.poisson(2).logpmf(x)),
            lambda x: st.poisson(2).logcdf(x - 1),
            (pt.random.poisson(2), 3),
        ),
        (pt.gt, st.poisson(2).logsf, st.poisson(2).logcdf, (pt.random.poisson(2), 3)),
        (pt.le, st.poisson(2).logcdf, st.poisson(2).logsf, (pt.random.poisson(2), 3)),
        (
            pt.lt,
            st.poisson(2).logsf,
            st.poisson(2).logcdf,
            (3, pt.random.poisson(2)),
        ),
        (pt.ge, st.poisson(2).logcdf, st.poisson(2).logsf, (3, pt.random.poisson(2))),
        (
            pt.gt,
            lambda x: st.poisson(2).logcdf(x - 1),
            lambda x: np.logaddexp(st.poisson(2).logsf(x), st.poisson(2).logpmf(x)),
            (3, pt.random.poisson(2)),
        ),
        (
            pt.le,
            lambda x: np.logaddexp(st.poisson(2).logsf(x), st.poisson(2).logpmf(x)),
            lambda x: st.poisson(2).logcdf(x - 1),
            (3, pt.random.poisson(2)),
        ),
    ],
)
def test_discrete_rv_comparison_bitwise(inputs, comparison_op, exp_logp_true, exp_logp_false):
    cens_x_rv = comparison_op(*inputs)

    cens_x_vv = cens_x_rv.clone()

    logprob = logp(cens_x_rv, cens_x_vv)
    assert_no_rvs(logprob)

    logp_fn = pytensor.function([cens_x_vv], logprob)

    assert np.isclose(logp_fn(1), exp_logp_true(3))
    assert np.isclose(logp_fn(0), exp_logp_false(3))

    bitwise_rv = pt.bitwise_not(comparison_op(*inputs))
    bitwise_vv = bitwise_rv.clone()

    logprob_not = logp(bitwise_rv, bitwise_vv)
    assert_no_rvs(logprob_not)

    logp_fn_not = pytensor.function([bitwise_vv], logprob_not)

    assert np.isclose(logp_fn_not(1), exp_logp_false(3))
    assert np.isclose(logp_fn_not(0), exp_logp_true(3))


def test_potentially_measurable_operand():
    x_rv = pt.random.normal(2)
    z_rv = pt.random.normal(x_rv)
    y_rv = pt.lt(x_rv, z_rv)

    y_vv = y_rv.clone()
    z_vv = z_rv.clone()

    logprob = conditional_logp({z_rv: z_vv, y_rv: y_vv})[y_vv]
    assert_no_rvs(logprob)

    fn = function([z_vv, y_vv], logprob)
    z_vv_test = 0.5
    y_vv_test = True
    np.testing.assert_array_almost_equal(
        fn(z_vv_test, y_vv_test),
        st.norm(2, 1).logcdf(z_vv_test),
    )

    with pytest.raises(
        NotImplementedError,
        match="Logprob method not implemented",
    ):
        logp(y_rv, y_vv).eval({y_vv: y_vv_test})


def test_comparison_invalid_broadcast():
    x_rv = pt.random.normal(0.5, 1, size=(3,))

    const = np.array([[0.1], [0.2], [-0.1]])
    y_rv_invalid = pt.gt(x_rv, const)

    y_vv_invalid = y_rv_invalid.clone()

    with pytest.raises(NotImplementedError, match="Logprob method not implemented for"):
        logp(y_rv_invalid, y_vv_invalid)
