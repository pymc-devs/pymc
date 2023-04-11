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

from pymc import logp
from pymc.testing import assert_no_rvs


@pytest.mark.parametrize(
    "comparison_op, exp_logp_true, exp_logp_false",
    [
        (pt.lt, st.norm(0, 1).logcdf, st.norm(0, 1).logsf),
        (pt.gt, st.norm(0, 1).logsf, st.norm(0, 1).logcdf),
    ],
)
def test_continuous_rv_comparison(comparison_op, exp_logp_true, exp_logp_false):
    x_rv = pt.random.normal(0, 1)
    comp_x_rv = comparison_op(x_rv, 0.5)

    comp_x_vv = comp_x_rv.clone()

    logprob = logp(comp_x_rv, comp_x_vv)
    assert_no_rvs(logprob)

    logp_fn = pytensor.function([comp_x_vv], logprob)

    assert np.isclose(logp_fn(0), exp_logp_false(0.5))
    assert np.isclose(logp_fn(1), exp_logp_true(0.5))


@pytest.mark.parametrize(
    "comparison_op, exp_logp_true, exp_logp_false",
    [
        (
            pt.lt,
            st.poisson(2).logcdf,
            lambda x: np.logaddexp(st.poisson(2).logsf(x), st.poisson(2).logpmf(x)),
        ),
        (
            pt.gt,
            lambda x: np.logaddexp(st.poisson(2).logsf(x), st.poisson(2).logpmf(x)),
            st.poisson(2).logcdf,
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
