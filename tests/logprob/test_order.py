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
import scipy.stats as st

import pymc as pm

from pymc import logp
from pymc.testing import assert_no_rvs


def test_max():
    x = pt.random.normal(0, 1, size=(3,))
    x.name = "x"
    x_max = pt.max(x, axis=-1)
    x_max_value = pt.vector("x_max_value")
    x_max_logprob = logp(x_max, x_max_value)

    assert_no_rvs(x_max_logprob)


def test_argmax():
    x = pt.random.normal(0, 1, size=(3,))
    x.name = "x"
    x_max = pt.argmax(x, axis=-1)
    x_max_value = pt.vector("x_max_value")

    with pytest.raises(RuntimeError, match=re.escape("Logprob method not implemented for Argmax")):
        x_max_logprob = logp(x_max, x_max_value)


def test_max_non_iid_fails():
    x = pm.Normal.dist([0, 1, 2, 3, 4], 1, shape=(5,))
    x.name = "x"
    x_max = pt.max(x, axis=-1)
    x_max_value = pt.vector("x_max_value")
    with pytest.raises(RuntimeError, match=re.escape("Logprob method not implemented")):
        x_max_logprob = logp(x_max, x_max_value)


def test_max_non_rv_fails():
    x = pt.exp(pt.random.normal(0, 1, size=(3,)))
    x.name = "x"
    x_max = pt.max(x, axis=-1)
    x_max_value = pt.vector("x_max_value")
    with pytest.raises(RuntimeError, match=re.escape("Logprob method not implemented")):
        x_max_logprob = logp(x_max, x_max_value)


def test_max_categorical():
    x = pm.Categorical.dist([1, 1, 1, 1], shape=(5,))
    x.name = "x"
    x_max = pt.max(x, axis=-1)
    x_max_value = pt.vector("x_max_value")
    with pytest.raises(RuntimeError, match=re.escape("Logprob method not implemented")):
        x_max_logprob = logp(x_max, x_max_value)


def test_max_logprob():
    x = pt.random.uniform(0, 1, size=(3,))
    x.name = "x"
    x_max = pt.max(x, axis=-1)
    x_max_value = pt.scalar("x_max_value")
    x_max_logprob = logp(x_max, x_max_value)

    test_value = 0.85  # or a vector of your choice

    beta_rv = pt.random.beta(3, 1, name="beta")
    beta_vv = beta_rv.clone()
    beta_rv_logprob = logp(beta_rv, beta_vv)

    np.testing.assert_allclose(
        beta_rv_logprob.eval({beta_vv: test_value}), (x_max_logprob.eval({x_max_value: test_value}))
    )
