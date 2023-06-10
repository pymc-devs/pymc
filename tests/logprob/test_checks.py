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
import re

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.raise_op import Assert
from scipy import stats

from pymc.distributions import Dirichlet
from pymc.logprob.basic import conditional_logp
from tests.distributions.test_multivariate import dirichlet_logpdf


def test_specify_shape_logprob():
    # 1. Create graph using SpecifyShape
    # Use symbolic last dimension, so that SpecifyShape is not useless
    last_dim = pt.scalar(name="last_dim", dtype="int64")
    x_base = Dirichlet.dist(pt.ones((last_dim,)), shape=(5, last_dim))
    x_base.name = "x"
    x_rv = pt.specify_shape(x_base, shape=(5, 3))
    x_rv.name = "x"

    # 2. Request logp
    x_vv = x_rv.clone()
    [x_logp] = conditional_logp({x_rv: x_vv}).values()

    # 3. Test logp
    x_logp_fn = pytensor.function([last_dim, x_vv], x_logp)

    # 3.1 Test valid logp
    x_vv_test = stats.dirichlet(np.ones((3,))).rvs(size=(5,))
    np.testing.assert_array_almost_equal(
        x_logp_fn(last_dim=3, x=x_vv_test),
        dirichlet_logpdf(x_vv_test, np.ones((3,))),
    )

    # 3.2 Test shape error
    x_vv_test_invalid = stats.dirichlet(np.ones((1,))).rvs(size=(5,))
    with pytest.raises(TypeError, match=re.escape("not compatible with the data's ((5, 1))")):
        x_logp_fn(last_dim=1, x=x_vv_test_invalid)


def test_assert_logprob():
    rv = pt.random.normal()
    assert_op = Assert("Test assert")
    # Example: Add assert that rv must be positive
    assert_rv = assert_op(rv, rv > 0)
    assert_rv.name = "assert_rv"

    assert_vv = assert_rv.clone()
    assert_logp = conditional_logp({assert_rv: assert_vv})[assert_vv]

    # Check valid value is correct and doesn't raise
    # Since here the value to the rv satisfies the condition, no error is raised.
    valid_value = 3.0
    np.testing.assert_allclose(
        assert_logp.eval({assert_vv: valid_value}),
        stats.norm.logpdf(valid_value),
    )

    # Check invalid value
    # Since here the value to the rv is negative, an exception is raised as the condition is not met
    with pytest.raises(AssertionError, match="Test assert"):
        assert_logp.eval({assert_vv: -5.0})
