# Copyright (c) 2024, PyMC Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# tests/test_print_value.py

import pymc as pm
import pytensor.tensor as pt
from pymc.printing import print_value


def test_print_value_is_passthrough():
    """print_value should not change the variable's value."""
    x = pt.vector("x")
    x_printed = print_value(x, name="test_x")

    import pytensor
    import numpy as np

    f = pytensor.function([x], x_printed)
    result = f([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_print_value_default_name():
    """print_value should use var.name if no name is given."""
    x = pt.vector("my_var")
    x_printed = print_value(x)
    # The Print op's message should match the variable name
    assert x_printed.owner.op.message == "my_var"


def test_print_value_custom_name():
    """print_value should use the custom name when provided."""
    x = pt.vector("x")
    x_printed = print_value(x, name="custom_label")
    assert x_printed.owner.op.message == "custom_label"


def test_print_value_accessible_from_pm():
    """print_value should be accessible as pm.print_value."""
    assert hasattr(pm, "print_value")
