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
import pytest

from pytensor import tensor as pt

from pymc.logprob.basic import logp


@pytest.mark.parametrize("axis", (None, 0))
def test_sum_of_normals_logprob(axis):
    mu = pt.constant([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    sigma = pt.constant([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])

    x_rv = pt.random.normal(mu, sigma, name="x")
    x_sum = pt.sum(x_rv, axis=axis)
    x_sum_vv = pt.scalar("x_sum")

    sum_logp = logp(x_sum, x_sum_vv)

    ref_mu = pt.sum(mu, axis=axis)
    ref_sigma = pt.sqrt(pt.sum(pt.square(sigma), axis=axis))
    ref_rv = pt.random.normal(ref_mu, ref_sigma, name="ref")
    ref_vv = pt.scalar("ref_vv")
    ref_logp = logp(ref_rv, ref_vv)

    test_val = 0.5
    np.testing.assert_allclose(
        sum_logp.eval({x_sum_vv: test_val}),
        ref_logp.eval({ref_vv: test_val}),
    )


@pytest.mark.parametrize("axis", (None, 0))
def test_add_of_normals_logprob(axis):
    mu1 = pt.constant([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    sigma1 = pt.constant([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    mu2 = pt.constant([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
    sigma2 = pt.constant([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])

    x1_rv = pt.random.normal(mu1, sigma1, name="x1")
    x2_rv = pt.random.normal(mu2, sigma2, name="x2")
    x_add = pt.add(x1_rv, x2_rv)
    if axis is not None:
        x_add = pt.sum(x_add, axis=axis)
    x_add_vv = pt.scalar("x_add")

    add_logp = logp(x_add, x_add_vv)

    ref_mu = pt.add(mu1, mu2)
    ref_sigma = pt.sqrt(pt.square(sigma1) + pt.square(sigma2))
    if axis is not None:
        ref_mu = pt.sum(ref_mu, axis=axis)
        ref_sigma = pt.sqrt(pt.sum(pt.square(ref_sigma), axis=axis))
    ref_rv = pt.random.normal(ref_mu, ref_sigma, name="ref")
    ref_vv = pt.scalar("ref_vv")
    ref_logp = logp(ref_rv, ref_vv)

    test_val = 0.5
    np.testing.assert_allclose(
        add_logp.eval({x_add_vv: test_val}),
        ref_logp.eval({ref_vv: test_val}),
    )


def test_sub_of_normals_logprob():
    mu1 = pt.constant([1.0, 2.0, 3.0])
    sigma1 = pt.constant([1.0, 2.0, 3.0])
    mu2 = pt.constant([0.5, 1.0, 1.5])
    sigma2 = pt.constant([0.5, 1.0, 1.5])

    x1_rv = pt.random.normal(mu1, sigma1, name="x1")
    x2_rv = pt.random.normal(mu2, sigma2, name="x2")
    x_sub = pt.sub(x1_rv, x2_rv)
    x_sub_vv = pt.scalar("x_sub")

    sub_logp = logp(x_sub, x_sub_vv)

    ref_mu = mu1 - mu2
    ref_sigma = pt.sqrt(pt.square(sigma1) + pt.square(sigma2))
    ref_rv = pt.random.normal(ref_mu, ref_sigma, name="ref")
    ref_vv = pt.scalar("ref_vv")
    ref_logp = logp(ref_rv, ref_vv)

    test_val = 0.5
    np.testing.assert_allclose(
        sub_logp.eval({x_sub_vv: test_val}),
        ref_logp.eval({ref_vv: test_val}),
    )
