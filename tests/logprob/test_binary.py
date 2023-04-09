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
import scipy.stats as st

from pymc import logp
from pymc.testing import assert_no_rvs


def test_continuous_rv_comparison_lt():
    x_rv = pt.random.normal(0.5, 1)
    comp_x_rv = pt.lt(x_rv, 0.5)

    comp_x_vv = comp_x_rv.clone()
    comp_x_vv.tag.test_value = 0

    logprob = logp(comp_x_rv, comp_x_vv)
    assert_no_rvs(logprob)

    logp_fn = pytensor.function([comp_x_vv], logprob)
    ref_scipy = st.norm(0.5, 1)

    assert np.isclose(logp_fn(0), ref_scipy.logcdf(0.5))
    assert np.isclose(logp_fn(1), ref_scipy.logsf(0.5))


def test_continuous_rv_comparison_gt():
    x_rv = pt.random.normal(0.5, 1)
    comp_x_rv = pt.gt(x_rv, 0.5)

    comp_x_vv = comp_x_rv.clone()
    comp_x_vv.tag.test_value = 0

    logprob = logp(comp_x_rv, comp_x_vv)
    assert_no_rvs(logprob)

    logp_fn = pytensor.function([comp_x_vv], logprob)
    ref_scipy = st.norm(0.5, 1)

    assert np.isclose(logp_fn(0), ref_scipy.logsf(0.5))
    assert np.isclose(logp_fn(1), ref_scipy.logcdf(0.5))


def test_discrete_rv_comparison():
    x_rv = pt.random.poisson(2)
    cens_x_rv = pt.lt(x_rv, 3)

    cens_x_vv = cens_x_rv.clone()

    logprob = logp(cens_x_rv, cens_x_vv)
    assert_no_rvs(logprob)

    logp_fn = pytensor.function([cens_x_vv], logprob)
    ref_scipy = st.poisson(2)

    assert np.isclose(logp_fn(1), ref_scipy.logcdf(3))
    assert np.isclose(logp_fn(0), np.logaddexp(ref_scipy.logsf(3), ref_scipy.logpmf(3)))
