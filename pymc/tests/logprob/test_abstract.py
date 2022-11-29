#   Copyright 2022- The PyMC Developers
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

import aesara.tensor as at
import pytest

from aesara.scalar import Exp, exp
from aesara.tensor.random.basic import NormalRV

from pymc.logprob.abstract import (
    MeasurableElemwise,
    MeasurableVariable,
    UnmeasurableVariable,
    _get_measurable_outputs,
    assign_custom_measurable_outputs,
    noop_measurable_outputs_fn,
)


def assert_equal_hash(classA, classB):
    assert hash(classA) == hash(classA.id_obj)
    assert hash(classB) == hash(classB.id_obj)
    assert classA == classB
    assert hash(classA) == hash(classB)


@pytest.mark.parametrize(
    "op, id_obj, class_dict",
    [
        (None, None, UnmeasurableVariable.__dict__),
        (None, (1, 2), UnmeasurableVariable.__dict__),
        (
            NormalRV,
            (NormalRV, noop_measurable_outputs_fn),
            UnmeasurableVariable.__dict__,
        ),
    ],
)
def test_unmeasurable_variable_class(op, id_obj, class_dict):
    A_dict = class_dict.copy()
    B_dict = class_dict.copy()

    if id_obj is not None:
        A_dict["id_obj"] = id_obj
        B_dict["id_obj"] = id_obj

    if op is None:
        parent_classes = (UnmeasurableVariable,)
    else:
        parent_classes = (op, UnmeasurableVariable)

    A = type("A", parent_classes, A_dict)
    B = type("B", parent_classes, B_dict)

    assert_equal_hash(A, B)


def test_unmeasurable_meta_hash_reassignment():
    A_dict = UnmeasurableVariable.__dict__.copy()
    B_dict = UnmeasurableVariable.__dict__.copy()

    A_dict["id_obj"] = (1, 2)
    B_dict["id_obj"] = (1, 3)

    A = type("A", (UnmeasurableVariable,), A_dict)
    B = type("B", (UnmeasurableVariable,), B_dict)

    assert A != B
    assert hash(A) != hash(B)

    A.id_obj = (1, 3)

    assert_equal_hash(A, B)


def test_assign_custom_measurable_outputs():
    srng = at.random.RandomStream(seed=2320)

    X_rv = srng.normal(-10.0, 0.1, name="X")
    Y_rv = srng.normal(10.0, 0.1, name="Y")

    # manually checking assign_custom_measurable_outputs
    unmeasurable_X = assign_custom_measurable_outputs(X_rv.owner).op
    unmeasurable_Y = assign_custom_measurable_outputs(Y_rv.owner).op

    assert_equal_hash(unmeasurable_X.__class__, unmeasurable_Y.__class__)
    assert unmeasurable_X.__class__.__name__.startswith("Unmeasurable")
    assert unmeasurable_X.__class__ in _get_measurable_outputs.registry

    # passing unmeasurable_X into assign_custom_measurable_outputs does nothing

    unmeas_X_rv = unmeasurable_X(-5, 0.1, name="unmeas_X")

    unmeasurable_X2_node = assign_custom_measurable_outputs(unmeas_X_rv.owner)
    unmeasurable_X2 = unmeasurable_X2_node.op

    assert unmeasurable_X2_node == unmeas_X_rv.owner
    assert_equal_hash(unmeasurable_X.__class__, unmeasurable_X2.__class__)

    with pytest.raises(ValueError):
        assign_custom_measurable_outputs(unmeas_X_rv.owner, lambda x: x)


def test_measurable_elemwise():
    # Default does not accept any scalar_op
    with pytest.raises(TypeError, match=re.escape("scalar_op exp is not valid")):
        MeasurableElemwise(exp)

    class TestMeasurableElemwise(MeasurableElemwise):
        valid_scalar_types = (Exp,)

    measurable_exp_op = TestMeasurableElemwise(scalar_op=exp)
    measurable_exp = measurable_exp_op(0.0)
    assert isinstance(measurable_exp.owner.op, MeasurableVariable)
