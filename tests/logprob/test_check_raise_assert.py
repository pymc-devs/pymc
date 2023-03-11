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

import pytensor.tensor as pt
import pytest

from pytensor.raise_op import Assert

from pymc.logprob.joint_logprob import factorized_joint_logprob


def test_assert_logprob():
    rv = pt.random.normal()
    assert_op = Assert("Test assert")
    # Example: Add assert that rv must be positive
    assert_rv = assert_op(rv > 0, rv)
    assert_rv.name = "assert_rv"

    assert_vv = assert_rv.clone()
    assert_logp = factorized_joint_logprob({assert_rv: assert_vv})[assert_vv]

    # Check valid value is correct and doesn't raise
    # Since here the value to the rv satisfies the condition, no error is raised.
    valid_value = 3.0
    with pytest.raises(AssertionError, match="Test assert"):
        assert_logp.eval({assert_vv: valid_value})

    # Check invalid value
    # Since here the value to the rv is negative, an exception is raised as the condition is not met
    with pytest.raises(AssertionError, match="Test assert"):
        assert_logp.eval({assert_vv: -5.0})
