#   Copyright 2024 The PyMC Developers
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
import scipy.stats as st

from pymc import logp
from pymc.logprob.basic import conditional_logp
from pymc.testing import assert_no_rvs


@pytest.mark.parametrize(
    "size, axis",
    [
        (10, None),
        (10, 0),
        ((2, 10), 0),
        ((2, 10), 1),
        ((3, 2, 10), 0),
        ((3, 2, 10), 1),
        ((3, 2, 10), 2),
    ],
)
def test_normal_cumsum(size, axis):
    rv = pt.random.normal(0, 1, size=size).cumsum(axis)
    vv = rv.clone()
    logprob = logp(rv, vv)
    assert_no_rvs(logprob)

    assert np.isclose(
        st.norm(0, 1).logpdf(np.ones(size)).sum(),
        logprob.eval({vv: np.ones(size).cumsum(axis)}).sum(),
    )


@pytest.mark.parametrize(
    "size, axis",
    [
        (10, None),
        (10, 0),
        ((2, 10), 0),
        ((2, 10), 1),
        ((3, 2, 10), 0),
        ((3, 2, 10), 1),
        ((3, 2, 10), 2),
    ],
)
def test_bernoulli_cumsum(size, axis):
    rv = pt.random.bernoulli(0.9, size=size).cumsum(axis)
    vv = rv.clone()
    logprob = logp(rv, vv)
    assert_no_rvs(logprob)

    assert np.isclose(
        st.bernoulli(0.9).logpmf(np.ones(size)).sum(),
        logprob.eval({vv: np.ones(size, int).cumsum(axis)}).sum(),
    )


def test_destructive_cumsum_fails():
    """Test that a cumsum that mixes dimensions fails"""
    x_rv = pt.random.normal(size=(2, 2, 2)).cumsum()
    x_vv = x_rv.clone()
    with pytest.raises(RuntimeError, match="could not be derived"):
        conditional_logp({x_rv: x_vv})


def test_deterministic_cumsum():
    """Test that deterministic cumsum is not affected"""
    x_rv = pt.random.normal(1, 1, size=5)
    cumsum_x_rv = pt.cumsum(x_rv)
    y_rv = pt.random.normal(cumsum_x_rv, 1)

    x_vv = x_rv.clone()
    y_vv = y_rv.clone()

    logp = conditional_logp({x_rv: x_vv, y_rv: y_vv})
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])
    assert_no_rvs(logp_combined)

    logp_fn = pytensor.function([x_vv, y_vv], logp_combined)
    assert np.isclose(
        logp_fn(np.ones(5), np.arange(5) + 1).sum(),
        st.norm(1, 1).logpdf(1) * 10,
    )
