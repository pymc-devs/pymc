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
from pymc.logprob import factorized_joint_logprob
from pymc.testing import assert_no_rvs


@pytest.mark.parametrize(
    "comparison_op, exp_logp_true, exp_logp_false",
    [
        ((pt.lt, pt.le), "logcdf", "logsf"),
        ((pt.gt, pt.ge), "logsf", "logcdf"),
    ],
)
def test_continuous_rv_comparison(comparison_op, exp_logp_true, exp_logp_false):
    x_rv = pt.random.normal(0, 1)
    for op in comparison_op:
        comp_x_rv = op(x_rv, 0.5)

        comp_x_vv = comp_x_rv.clone()

        logprob = logp(comp_x_rv, comp_x_vv)
        assert_no_rvs(logprob)

        logp_fn = pytensor.function([comp_x_vv], logprob)
        ref_scipy = st.norm(0, 1)

        assert np.isclose(logp_fn(0), getattr(ref_scipy, exp_logp_false)(0.5))
        assert np.isclose(logp_fn(1), getattr(ref_scipy, exp_logp_true)(0.5))


@pytest.mark.parametrize(
    "comparison_op, exp_logp_true, exp_logp_false",
    [
        (
            pt.lt,
            lambda x: st.poisson(2).logcdf(x - 1),
            lambda x: np.logaddexp(st.poisson(2).logsf(x), st.poisson(2).logpmf(x)),
        ),
        (
            pt.ge,
            lambda x: np.logaddexp(st.poisson(2).logsf(x), st.poisson(2).logpmf(x)),
            lambda x: st.poisson(2).logcdf(x - 1),
        ),
        (
            pt.gt,
            st.poisson(2).logsf,
            st.poisson(2).logcdf,
        ),
        (
            pt.le,
            st.poisson(2).logcdf,
            st.poisson(2).logsf,
        ),
    ],
)
def test_discrete_rv_comparison(comparison_op, exp_logp_true, exp_logp_false):
    x_rv = pt.random.poisson(2)
    cens_x_rv = comparison_op(x_rv, 3)

    cens_x_vv = cens_x_rv.clone()

    logprob = logp(cens_x_rv, cens_x_vv)
    assert_no_rvs(logprob)

    logp_fn = pytensor.function([cens_x_vv], logprob)

    assert np.isclose(logp_fn(1), exp_logp_true(3))
    assert np.isclose(logp_fn(0), exp_logp_false(3))


def test_potentially_measurable_operand():
    x_rv = pt.random.normal(2)
    z_rv = pt.random.normal(x_rv)
    y_rv = pt.lt(x_rv, z_rv)

    y_vv = y_rv.clone()
    z_vv = z_rv.clone()

    logprob = factorized_joint_logprob({z_rv: z_vv, y_rv: y_vv})[y_vv]
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
