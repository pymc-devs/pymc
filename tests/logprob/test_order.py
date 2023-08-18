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

import re

import numpy as np
import pytensor.tensor as pt
import pytest

import pymc as pm

from pymc import logp
from pymc.testing import assert_no_rvs


def test_argmax():
    """Test whether the logprob for ```pt.argmax``` is correctly rejected"""
    x = pt.random.normal(0, 1, size=(3,))
    x.name = "x"
    x_max = pt.argmax(x, axis=-1)
    x_max_value = pt.vector("x_max_value")

    with pytest.raises(RuntimeError, match=re.escape("Logprob method not implemented for Argmax")):
        x_max_logprob = logp(x_max, x_max_value)


@pytest.mark.parametrize(
    "pt_op",
    [
        pt.max,
        pt.min,
    ],
)
def test_non_iid_fails(pt_op):
    """Test whether the logprob for ```pt.max``` or ```pt.min``` for non i.i.d is correctly rejected"""
    x = pm.Normal.dist([0, 1, 2, 3, 4], 1, shape=(5,))
    x.name = "x"
    x_m = pt_op(x, axis=-1)
    x_m_value = pt.vector("x_value")
    with pytest.raises(RuntimeError, match=re.escape("Logprob method not implemented")):
        x_max_logprob = logp(x_m, x_m_value)


@pytest.mark.parametrize(
    "pt_op",
    [
        pt.max,
        pt.min,
    ],
)
def test_non_rv_fails(pt_op):
    """Test whether the logprob for ```pt.max``` for non-RVs is correctly rejected"""
    x = pt.exp(pt.random.beta(0, 1, size=(3,)))
    x.name = "x"
    x_m = pt_op(x, axis=-1)
    x_m_value = pt.vector("x_value")
    with pytest.raises(RuntimeError, match=re.escape("Logprob method not implemented")):
        x_max_logprob = logp(x_m, x_m_value)


@pytest.mark.parametrize(
    "pt_op",
    [
        pt.max,
        pt.min,
    ],
)
def test_multivariate_rv_fails(pt_op):
    _alpha = pt.scalar()
    _k = pt.iscalar()
    x = pm.StickBreakingWeights.dist(_alpha, _k)
    x.name = "x"
    x_m = pt_op(x, axis=-1)
    x_m_value = pt.vector("x_value")
    with pytest.raises(RuntimeError, match=re.escape("Logprob method not implemented")):
        x_max_logprob = logp(x_m, x_m_value)


@pytest.mark.parametrize(
    "pt_op",
    [
        pt.max,
        pt.min,
    ],
)
def test_categorical(pt_op):
    """Test whether the logprob for ```pt.max``` for unsupported distributions is correctly rejected"""
    x = pm.Categorical.dist([1, 1, 1, 1], shape=(5,))
    x.name = "x"
    x_m = pt_op(x, axis=-1)
    x_m_value = pt.vector("x_value")
    with pytest.raises(RuntimeError, match=re.escape("Logprob method not implemented")):
        x_max_logprob = logp(x_m, x_m_value)


@pytest.mark.parametrize(
    "pt_op",
    [
        pt.max,
        pt.min,
    ],
)
def test_non_supp_axis(pt_op):
    """Test whether the logprob for ```pt.max``` for unsupported axis is correctly rejected"""
    x = pt.random.normal(0, 1, size=(3, 3))
    x.name = "x"
    x_m = pt_op(x, axis=-1)
    x_m_value = pt.vector("x_value")
    with pytest.raises(RuntimeError, match=re.escape("Logprob method not implemented")):
        x_max_logprob = logp(x_m, x_m_value)


@pytest.mark.parametrize(
    "shape, value, axis",
    [
        (3, 0.85, -1),
        (3, 0.01, 0),
        (2, 0.2, None),
        (4, 0.5, 0),
        ((3, 4), 0.9, None),
        ((3, 4), 0.75, (1, 0)),
    ],
)
def test_max_logprob(shape, value, axis):
    """Test whether the logprob for ```pt.max``` produces the corrected

    The fact that order statistics of i.i.d. uniform RVs ~ Beta is used here:
        U_1, \\dots, U_n \\stackrel{\text{i.i.d.}}{\\sim} \text{Uniform}(0, 1) \\Rightarrow U_{(k)} \\sim \text{Beta}(k, n + 1- k)
    for all 1<=k<=n
    """
    x = pt.random.uniform(0, 1, size=shape)
    x.name = "x"
    x_max = pt.max(x, axis=axis)
    x_max_value = pt.scalar("x_max_value")
    x_max_logprob = logp(x_max, x_max_value)

    assert_no_rvs(x_max_logprob)

    test_value = value

    n = np.prod(shape)
    beta_rv = pt.random.beta(n, 1, name="beta")
    beta_vv = beta_rv.clone()
    beta_rv_logprob = logp(beta_rv, beta_vv)

    np.testing.assert_allclose(
        beta_rv_logprob.eval({beta_vv: test_value}),
        (x_max_logprob.eval({x_max_value: test_value})),
        rtol=1e-06,
    )


@pytest.mark.parametrize(
    "shape, value, axis",
    [
        (3, 0.85, -1),
        (3, 0.01, 0),
        (2, 0.2, None),
        (4, 0.5, 0),
        ((3, 4), 0.9, None),
        ((3, 4), 0.75, (1, 0)),
    ],
)
def test_min_logprob(shape, value, axis):
    """Test whether the logprob for ```pt.mix``` produces the corrected
    The fact that order statistics of i.i.d. uniform RVs ~ Beta is used here:
        U_1, \\dots, U_n \\stackrel{\text{i.i.d.}}{\\sim} \text{Uniform}(0, 1) \\Rightarrow U_{(k)} \\sim \text{Beta}(k, n + 1- k)
    for all 1<=k<=n
    """
    x = pt.random.uniform(0, 1, size=shape)
    x.name = "x"
    x_min = pt.min(x, axis=axis)
    x_min_value = pt.scalar("x_min_value")
    x_min_logprob = logp(x_min, x_min_value)

    assert_no_rvs(x_min_logprob)

    test_value = value

    n = np.prod(shape)
    beta_rv = pt.random.beta(1, n, name="beta")
    beta_vv = beta_rv.clone()
    beta_rv_logprob = logp(beta_rv, beta_vv)

    np.testing.assert_allclose(
        beta_rv_logprob.eval({beta_vv: test_value}),
        (x_min_logprob.eval({x_min_value: test_value})),
        rtol=1e-06,
    )


def test_min_non_mul_elemwise_fails():
    """Test whether the logprob for ```pt.min``` for non-mul elemwise RVs is rejected correctly"""
    x = pt.log(pt.random.beta(0, 1, size=(3,)))
    x.name = "x"
    x_min = pt.min(x, axis=-1)
    x_min_value = pt.vector("x_min_value")
    with pytest.raises(RuntimeError, match=re.escape("Logprob method not implemented")):
        x_min_logprob = logp(x_min, x_min_value)
