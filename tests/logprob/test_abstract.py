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
import pytensor.tensor as pt
import pytest
import scipy.stats.distributions as sp

from pytensor.scalar import Exp, exp
from pytensor.tensor.random.basic import NormalRV

import pymc as pm

from pymc.logprob.abstract import MeasurableElemwise, MeasurableVariable, _logcdf_helper
from pymc.logprob.basic import logcdf


def assert_equal_hash(classA, classB):
    assert hash(classA) == hash(classA.id_obj)
    assert hash(classB) == hash(classB.id_obj)
    assert classA == classB
    assert hash(classA) == hash(classB)


def test_measurable_elemwise():
    # Default does not accept any scalar_op
    with pytest.raises(TypeError, match=re.escape("scalar_op exp is not valid")):
        MeasurableElemwise(exp)

    class TestMeasurableElemwise(MeasurableElemwise):
        valid_scalar_types = (Exp,)

    measurable_exp_op = TestMeasurableElemwise(scalar_op=exp)
    measurable_exp = measurable_exp_op(0.0)
    assert isinstance(measurable_exp.owner.op, MeasurableVariable)


def test_logcdf_helper():
    value = pt.vector("value")
    x = pm.Normal.dist(0, 1)

    x_logcdf = _logcdf_helper(x, value)
    np.testing.assert_almost_equal(x_logcdf.eval({value: [0, 1]}), sp.norm(0, 1).logcdf([0, 1]))

    x_logcdf = _logcdf_helper(x, [0, 1])
    np.testing.assert_almost_equal(x_logcdf.eval(), sp.norm(0, 1).logcdf([0, 1]))


def test_logcdf_transformed_argument():
    with pm.Model() as m:
        sigma = pm.HalfFlat("sigma")
        x = pm.Normal("x", 0, sigma)
        pm.Potential("norm_term", -logcdf(x, 1.0))

    sigma_value_log = -1.0
    sigma_value = np.exp(sigma_value_log)
    x_value = 0.5

    observed = m.compile_logp(jacobian=False)({"sigma_log__": sigma_value_log, "x": x_value})
    expected = pm.logp(
        pm.TruncatedNormal.dist(0, sigma_value, lower=None, upper=1.0), x_value
    ).eval()
    assert np.isclose(observed, expected)
